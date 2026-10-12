"""Offline checks only; real pilot/training runs on Kaggle."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from src.common import digest, read_csv, save_array, write_csv, write_json
from src.experiment import evaluate, freeze, tune
from src.metrics import classification
from src.training import LowLightDataset, train
from scripts.verify_results import verify


def main():
    torch.set_num_threads(2)
    config = json.loads(Path('configs/default.json').read_text())
    config['cnn'].update(epochs=2, batch_size=2, patience=5)
    config['training'] = {'regime': 'mixed_lowlight', 'clean_probability': 0.5,
                          'train_noise_seed': 4101, 'validation_noise_seed': 9101,
                          'clean_min_macro_f1': 0.8}
    config['bootstrap_repetitions'] = 5
    with TemporaryDirectory() as folder:
        root = Path(folder)
        rows = []
        for split in ['train', 'validation', 'test']:
            for i in range(2):
                path = f'{split}_{i}.npy'
                save_array(root / path, np.full((64, 64, 3), 0.6, np.float32))
                rows.append({'sample_id': f'{split}/{i}', 'class_id': i, 'split': split,
                             'group_id': f'{split}/{i}', 'source_id': f'{split}/{i}', 'crop_path': path})
        manifest = root / 'data/manifests/gtsrb_samples.csv'
        write_csv(manifest, rows)
        write_json(manifest.with_suffix('.json'), {'sha256': digest(manifest)})
        write_json(manifest.parent / 'audit.json', {'status': 'completed'})
        dataset = LowLightDataset(root, rows, config, 11, 'mixed')
        observed = []
        for epoch in range(20):
            dataset.epoch = epoch
            a, label = dataset[0]
            torch.testing.assert_close(a, dataset[0][0])
            assert label == 0 and torch.isfinite(a).all() and 0 <= a.min() <= a.max() <= 1
            observed.append(a.numpy())
        assert any(np.array_equal(x, np.full_like(x, 0.6)) for x in observed)
        assert any(not np.array_equal(observed[0], x) for x in observed[1:])
        fixed = LowLightDataset(root, rows, config, 11, 'severe')
        a = fixed[0][0]
        fixed.epoch = 100
        torch.testing.assert_close(a, fixed[0][0])
        other = deepcopy(config)
        other['training']['validation_noise_seed'] += 1
        assert not torch.equal(a, LowLightDataset(root, rows, other, 11, 'severe')[0][0])
        # Epoch 2 has higher mixed F1 but violates the clean guardrail: retain epoch 1.
        values = iter([0.1, 0.9, 0.4, 0.5, 0.6, 0.1, 0.7, 0.99, 0.99, 0.99])
        def score(labels, predictions):
            result = classification(labels, predictions)
            result['macro_f1'] = next(values)
            return result
        with patch('src.training.classification', side_effect=score):
            checkpoint = train(root, config, 11, torch.device('cpu'), root / 'train')
        saved = torch.load(checkpoint, weights_only=True)
        assert saved['epoch'] == 1 and abs(saved['macro_f1'] - 0.6) < 1e-12
        assert saved['training'] == config['training']
        assert len(read_csv(root / 'train/learning_11.csv')) == 10
        assert train(root, config, 11, torch.device('cpu'), root / 'train') == checkpoint
        wrong = deepcopy(config)
        wrong['training']['regime'] = 'clean'
        try:
            tune(root, wrong, [checkpoint], torch.device('cpu'), root / 'wrong', methods=['identity'])
        except ValueError as error:
            assert 'regime' in str(error)
        else:
            raise AssertionError('Accepted mismatched training regime')
        config['noise_seeds'] = [101]
        tune(root, config, [checkpoint], torch.device('cpu'), root / 'validation', methods=['identity'])
        protocol = freeze(root, root / 'validation/selection.json', root / 'protocol.json', pilot=True)
        assert protocol['protocol'] == 'mixed_lowlight' and len(protocol['runs']) == 4
        result = evaluate(root, root / 'protocol.json', root / 'evaluation', torch.device('cpu'))
        assert all(r['protocol'] == 'mixed_lowlight' for r in result)
        write_csv(root / 'evaluation/aggregate.csv', [
            {'condition': r['condition'], 'method': r['method'], 'n_training_seeds': 1,
             **{k + '_mean': r[k] for k in ['accuracy', 'macro_f1', 'weighted_f1']},
             **{k + '_std': 0 for k in ['accuracy', 'macro_f1', 'weighted_f1']}} for r in result])
        assert verify(root, root / 'evaluation')['predictions_checked'] == 8
        prediction_file = root / 'evaluation/predictions' / f"{result[0]['run_id']}.csv"
        broken = read_csv(prediction_file)
        broken[0]['class_id'] = 42
        write_csv(prediction_file, broken)
        try:
            verify(root, root / 'evaluation')
        except AssertionError:
            pass
        else:
            raise AssertionError('Accepted corrupted prediction labels')
        # All epochs fail guardrail: preserve the failed metadata and refuse test.
        with patch('src.training.classification', return_value=classification([0, 1], [42, 42])):
            try:
                train(root, config, 22, torch.device('cpu'), root / 'failed')
            except ValueError as error:
                assert 'guardrail' in str(error)
            else:
                raise AssertionError('Accepted run with no eligible checkpoint')
        assert json.loads((root / 'failed/train_22.json').read_text())['status'] == 'failed'
    print('PASS: mixed sampling/replay, validation noise, guarded checkpoint selection, regime labels and failure handling')


if __name__ == '__main__':
    main()
