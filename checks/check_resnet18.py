"""Offline classifier checks: python -m checks.check_resnet18."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from src.common import digest, read_csv, save_array, write_csv, write_json
from src.experiment import load_cnn, predict
from src.metrics import classification
from src.model import cnn_config, make_cnn
from src.training import CropDataset, train


def rejects(action, message):
    try:
        action()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f'Accepted invalid input: {message}')


def main():
    torch.set_num_threads(2)
    torch.manual_seed(11)
    device = torch.device('cpu')
    config = json.loads(Path('configs/resnet18.json').read_text())
    cfg = config['cnn']
    images = torch.rand(2, 3, 64, 64)
    legacy = make_cnn().eval()
    assert legacy(images).shape == (2, 43)
    assert '0.weight' in legacy.state_dict()
    model = make_cnn(cfg).eval()
    expected_input = (F.interpolate(images, (224, 224), mode='bilinear', align_corners=False,
                                     antialias=True) - model.mean) / model.std
    observed = []
    hook = model.backbone.register_forward_pre_hook(lambda _, args: observed.append(args[0].detach()))
    with torch.no_grad():
        logits = model(images)
        expected = model.backbone(expected_input)
    hook.remove()
    assert logits.shape == (2, 43) and torch.isfinite(logits).all()
    torch.testing.assert_close(observed[0], expected_input)
    torch.testing.assert_close(logits, expected)
    model.train()
    before = [p.detach().clone() for p in [model.backbone.conv1.weight, model.backbone.fc.weight]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    loss = F.cross_entropy(model(images), torch.tensor([0, 19]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, [model.backbone.conv1.weight, model.backbone.fc.weight]))
    model.eval()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / 'cnn.pt'
        with patch('torch.hub.download_url_to_file', side_effect=AssertionError('Network used')):
            for candidate, settings in [(legacy, {}), (model, cfg)]:
                torch.save({'state_dict': candidate.state_dict(), 'manifest_sha256': 'test',
                            'cnn_config': settings}, checkpoint)
                restored, _ = load_cnn(checkpoint, 'test', device)
                with torch.no_grad():
                    torch.testing.assert_close(candidate(images), restored(images))
            rejects(lambda: load_cnn(checkpoint, 'other', device), 'manifest')
            rejects(lambda: load_cnn(checkpoint, 'test', device, {}), 'config')
        rows, cached = [], []
        for i in range(2):
            save_array(root / f'{i}.npy', images[i].permute(1, 2, 0).numpy())
            rows.append({'sample_id': str(i), 'group_id': str(i), 'source_id': str(i),
                         'class_id': i, 'crop_path': f'{i}.npy'})
            cached.append({'sample_id': str(i), 'path': f'{i}.npy', 'noise_seed': '',
                           'psnr': float('inf'), 'ssim': 1, 'latency_ms': 0})
        dataset = CropDataset(root, rows)
        with torch.no_grad():
            probabilities = model(torch.stack([dataset[i][0] for i in range(2)])).softmax(1)
        result = predict(root, rows, cached, model, config, device)
        np.testing.assert_allclose([r['true_probability'] for r in result],
                                   probabilities[range(2), range(2)].numpy(), rtol=1e-5)
        # Exercise actual training and safe skip using the inexpensive legacy model.
        training_config = json.loads(Path('configs/default.json').read_text())
        training_config['cnn'].update(epochs=1, batch_size=2)
        manifest_rows = [{**r, 'split': split, 'sample_id': f"{split}/{r['sample_id']}"}
                         for split in ['train', 'validation'] for r in rows]
        manifest = root / 'data/manifests/gtsrb_samples.csv'
        write_csv(manifest, manifest_rows)
        write_json(manifest.with_suffix('.json'), {'sha256': digest(manifest)})
        write_json(manifest.parent / 'audit.json', {'status': 'completed'})
        output = root / 'train'
        saved = train(root, training_config, 11, device, output)
        original_hash = digest(saved)
        assert train(root, training_config, 11, device, output) == saved
        assert digest(saved) == original_hash
        changed = deepcopy(training_config)
        changed['cnn']['learning_rate'] *= 2
        rejects(lambda: train(root, changed, 11, device, output), 'mismatch')
        rejects(lambda: train(root, training_config, 11, device, output, epochs=2), 'mismatch')
        record = json.loads((output / 'train_11.json').read_text())
        record['status'] = 'running'
        write_json(output / 'train_11.json', record)
        rejects(lambda: train(root, training_config, 11, device, output), 'incomplete')
        # Early stopping uses a significant-improvement anchor; checkpoint keeps the actual best.
        training_config['cnn'].update(epochs=20, patience=5, min_delta=0.001)
        for label, values, best_epoch, last_epoch in [
                ('plateau', [0.5, 0.501, 0.5008, 0.5009, 0.5007, 0.5006], 2, 6),
                ('cumulative', [0.5, 0.5006, 0.5012, 0.5015, 0.5016, 0.5017, 0.5018, 0.5019], 8, 8)]:
            metrics = classification([0, 1], [0, 1])
            mocked = [{**metrics, 'macro_f1': value} for value in values for _ in range(2)]
            with patch('src.training.classification', side_effect=mocked):
                saved = train(root, training_config, 11, device, root / label)
            checkpoint_data = torch.load(saved, map_location='cpu', weights_only=True)
            assert checkpoint_data['epoch'] == best_epoch
            assert checkpoint_data['macro_f1'] == max(values)
            assert int(read_csv(root / label / 'learning_11.csv')[-1]['epoch']) == last_epoch
        assert cnn_config({})['min_delta'] == 0.0
    for change in [{'architecture': 'unknown'}, {'normalization': 'none'}, {'input_size': 64},
                   {'weights': 'DEFAULT'}, {'optimizer': 'unknown'}, {'weight_decay': -1},
                   {'min_delta': -0.001}, {'min_delta': float('nan')}, {'min_delta': float('inf')}]:
        rejects(lambda: cnn_config({**cfg, **change}), 'nvalid' if 'architecture' not in change else 'Unknown')
    from scripts.run_resnet18_kaggle import select_configuration
    for trial_scores in [[0.95], [0.8, 0.94, 0.91]]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fake_train(root, config, seed, device, folder):
                calls.append(config['cnn']['learning_rate'])
                folder.mkdir(parents=True)
                score = trial_scores[len(calls) - 1]
                for name in ['cnn_11.pt', 'train_11.json', 'learning_11.csv']:
                    (folder / name).write_text(str(score))
                write_json(folder / 'validation_11.json',
                           dict(accuracy=score, macro_f1=score, weighted_f1=score))

            with patch('src.training.train', fake_train):
                selected = select_configuration(root, config, device)
            assert len(calls) == len(trial_scores)
            assert selected['cnn']['learning_rate'] == (1e-4 if len(trial_scores) == 1 else 3e-5)
            assert float((root / 'outputs/resnet18_A/train/cnn_11.pt').read_text()) == max(trial_scores)
    print('PASS: legacy/ResNet, preprocessing, gradients, offline round-trip, predict, config and safe skip')


if __name__ == '__main__':
    main()
