"""Temporary synthetic data checks orchestration, not GTSRB performance."""
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from src.common import digest, write_csv, write_json
from src.experiment import cache_images, evaluate, freeze, tune
from src.model import make_cnn
from src.reporting import report


def check(config_path):
    torch.set_num_threads(2)
    cfg = json.loads(Path(config_path).read_text())
    cfg['noise_seeds'] = [101]
    cfg['train_seeds'] = [11]
    cfg['bootstrap_repetitions'] = 10
    device = torch.device('cpu')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(5)
        rows = []
        for split in ['validation', 'test']:
            for label in range(43):
                image = rng.integers(0, 256, (64,64,3), dtype=np.uint8)
                path = root / f'{split}_{label}.png'
                Image.fromarray(image).save(path)
                rows.append({'sample_id': f'{split}/{label}', 'class_id': label, 'split': split,
                             'group_id': f'{split}/{label}', 'source_id': f'{split}/{label}', 'crop_path': path.name})
        manifest = root/'data/manifests/gtsrb_samples.csv'
        write_csv(manifest, rows)
        write_json(manifest.with_suffix('.json'), {'sha256': digest(manifest)})
        write_json(manifest.parent/'audit.json', {'status': 'completed'})
        checkpoint = root/'cnn.pt'
        torch.save({'state_dict': make_cnn(cfg['cnn']).state_dict(), 'manifest_sha256': digest(manifest),
                    'seed': 11, 'pilot': True, 'cnn_config': cfg['cnn']}, checkpoint)
        selection = root/'validation'
        tune(root,cfg,[checkpoint],device,selection,methods=['identity','gamma'])
        protocol = root/'protocol.json'
        try:
            freeze(root,selection/'selection.json',protocol)
        except ValueError as error:
            assert 'Pilot' in str(error)
        else:
            raise AssertionError('Pilot accepted as final protocol')
        freeze(root,selection/'selection.json',protocol,pilot=True)
        result = evaluate(root,protocol,root/'evaluation',device)
        assert len(result) == 7 and all(r['n_images']==43 for r in result)
        assert report(root/'evaluation', root).exists()
        assert all('weighted_f1' in r for r in result)
        from src.common import read_csv
        assert 'weighted_f1_mean' in read_csv(root/'evaluation/aggregate.csv')[0]
        assert (root/'evaluation/classifier_latency.json').exists()
        # Legacy report fallback must reconstruct Weighted-F1 without rewriting summary.
        write_csv(root/'evaluation/summary.csv', [{k: v for k, v in r.items() if k != 'weighted_f1'} for r in result])
        old_hash = digest(root/'evaluation/summary.csv')
        report(root/'evaluation', root)
        assert digest(root/'evaluation/summary.csv') == old_hash
        # A cached rerun must preserve all recognition and quality scores.
        repeated = evaluate(root,protocol,root/'evaluation',device)
        assert result == repeated
        quality = {(r['condition'], r['method'], r['noise_seed']):
                   (r['psnr_mean'], r['ssim_mean'],
                    [digest(root / p['output_path']) for p in read_csv(root/'evaluation/predictions'/f"{r['run_id']}.csv")])
                   for r in result if r['method'] in {'clean', 'identity'}}
        fixed_gamma = cache_images(root, [r for r in rows if r['split'] == 'test'], digest(manifest),
                                  cfg, 'medium', 101, 'gamma', {'gamma': 0.6}, device)
        quality['fixed_gamma'] = [(r['sha256'], float(r['psnr']), float(r['ssim'])) for r in fixed_gamma]
        # Changed checkpoint cannot be reused against a locked protocol.
        with checkpoint.open('ab') as stream:
            stream.write(b'changed')
        try:
            evaluate(root,protocol,root/'evaluation',device)
        except ValueError as error:
            assert 'checkpoint changed' in str(error)
        else:
            raise AssertionError('Changed checkpoint accepted')
    print('PASS: validation search, frozen protocol, evaluation, reporting, cache replay, checkpoint tamper detection')
    return quality


if __name__ == '__main__':
    baseline = check('configs/default.json')
    resnet = check('configs/resnet18.json')
    assert baseline == resnet, 'Classifier changed enhanced pixels or image metrics'
