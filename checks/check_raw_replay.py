"""Rebuild one fixed validation crop per class from raw GTSRB without modifying data."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.common import digest, read_csv, read_image, write_json
from src.data import subset_rows
from src.degradation import degrade


def main():
    root = Path.cwd()
    manifest = root / 'data/manifests/gtsrb_samples.csv'
    meta = json.loads(manifest.with_suffix('.json').read_text())
    assert digest(manifest) == meta['sha256']
    rows = subset_rows(read_csv(manifest), 'validation', 1, 42)
    selected = [next(r for r in rows if int(r['class_id']) == label) for label in range(43)]
    cfg = json.loads((root / 'configs/resnet18.json').read_text())
    for row in selected:
        source = root / row['source_path']
        assert digest(source) == row['source_sha256']
        with Image.open(source) as image:
            rebuilt = np.asarray(image.convert('RGB').crop(json.loads(row['bbox'])).resize(
                (64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        assert np.array_equal(rebuilt, read_image(root / row['crop_path'])), row['sample_id']
        for condition, parameters in cfg['conditions'].items():
            a = degrade(rebuilt, **parameters, sample_id=row['sample_id'], noise_seed=101, condition=condition)
            b = degrade(read_image(root / row['crop_path']), **parameters,
                        sample_id=row['sample_id'], noise_seed=101, condition=condition)
            assert np.array_equal(a, b)
    write_json(root / 'outputs/replay/verification.json', {
        'status': 'verified', 'n_samples': 43, 'split': 'validation', 'selection': 'first stable ID per class, seed 42',
        'source_sha256_checked': True, 'canonical_crop_pixel_match': 'exact',
        'degradation_replay_match': 'exact float32 at mild/medium/severe, noise seed 101',
        'manifest_sha256': digest(manifest), 'sample_ids': [r['sample_id'] for r in selected]})
    print('PASS: 43 raw sources reconstructed; canonical pixels and three degradation levels replay exactly')


if __name__ == '__main__':
    main()
