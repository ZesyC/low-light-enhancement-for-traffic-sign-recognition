"""Run from project root: python -m checks.check_pipeline (no test framework)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.common import read_image, save_array
from src.data import crop_box, split_tracks, quarantine_test_overlap
from src.degradation import degrade
from src.enhancement import classical
from src.experiment import choose_pipeline
from src.metrics import classification, image_metrics, paired_rows, paired_bootstrap
from src.model import make_cnn
from src.training import TRAIN_TRANSFORM


def main():
    torch.set_num_threads(2)
    config = json.loads(Path('configs/default.json').read_text())
    x = np.full((64, 64, 3), 0.5, np.float32)
    assert np.array_equal(degrade(x), x)
    assert np.allclose(degrade(x, gamma_dark=2), 0.25)
    assert np.allclose(classical(x, 'gamma', {'gamma': 0.5}), np.sqrt(0.5))
    a = degrade(x, **config['conditions']['medium'], sample_id='a')
    assert np.array_equal(a, degrade(x, **config['conditions']['medium'], sample_id='a'))
    assert not np.array_equal(a, degrade(x, **config['conditions']['medium'], sample_id='b'))
    assert not np.array_equal(a, degrade(x, **config['conditions']['medium'], sample_id='a', noise_seed=202))
    z = np.full((512, 512, 3), 0.2, np.float32)
    p = degrade(z, poisson_peak=100)
    assert abs(p.mean()-0.2) < 0.0003 and abs(p.var()-0.002) < 0.00003
    for image in [x, np.zeros_like(x), np.random.default_rng(1).random(x.shape).astype(np.float32), np.ones((1, 2, 3), np.float32)]:
        for method in ['identity', 'gamma', 'he', 'clahe', 'retinex']:
            result = classical(image, method)
            assert result.shape == image.shape and np.isfinite(result).all()
            assert 0 <= result.min() <= result.max() <= 1
    for kwargs in [{'gamma_dark': 0}, {'poisson_peak': 0}, {'gaussian_sigma': -1}, {'gamma_dark': float('nan')}]:
        try:
            degrade(x, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid parameters accepted')
    assert crop_box([0, 0, 9, 9], 10, 10) == [0, 0, 10, 10]
    assert crop_box([2, 2, 7, 7], 10, 10, padding=0) == [2, 2, 8, 8]
    rows = [{'class_id': c, 'group_id': f'{c}/{t}', 'split': 'train'} for c in range(43) for t in range(4) for _ in range(2)]
    split_tracks(rows)
    train_groups = {r['group_id'] for r in rows if r['split'] == 'train'}
    val_groups = {r['group_id'] for r in rows if r['split'] == 'validation'}
    assert not train_groups & val_groups
    assert len({r['class_id'] for r in rows if r['split'] == 'validation'}) == 43
    overlap = [{'split': s, 'group_id': g, 'pixel_sha256': h} for s,g,h in [('train','track','same'),('train','track','other'),('test','test','same')]]
    assert quarantine_test_overlap(overlap) == ['track']
    assert [r['split'] for r in overlap] == ['quarantine','quarantine','test']
    score = classification([0, 0, 1, 1], [0, 1, 1, 1])
    assert score['accuracy'] == 0.75 and np.isclose(score['macro_f1'], (2/3 + 4/5)/43)
    quality = image_metrics(x, x, config['ssim'])
    assert np.isinf(quality['psnr']) and quality['ssim'] == 1
    before = [{'sample_id': str(i), 'noise_seed': 101, 'class_id': i, 'prediction': 0, 'group_id': str(i)} for i in range(2)]
    after = [dict(r, prediction=r['class_id']) for r in before]
    ci = paired_bootstrap(before, after, repetitions=20)
    assert ci['accuracy']['delta'] == 50
    try:
        paired_rows(before, [dict(after[0], sample_id='wrong'), after[1]])
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong sample ID accepted')
    assert choose_pipeline([{'method': 'identity', 'macro_f1': 0.5, 'latency_ms': 0},
                            {'method': 'gamma', 'macro_f1': 0.504, 'latency_ms': 1}])['method'] == 'identity'
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'image.npy'
        save_array(path, a)
        assert np.array_equal(read_image(path), a)
    torch.manual_seed(11)
    model = make_cnn()
    inputs, labels = torch.rand(2, 3, 64, 64), torch.tensor([0, 42])
    output = model(inputs)
    assert output.shape == (2, 43)
    loss = torch.nn.functional.cross_entropy(output, labels)
    weight = model[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters())
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(weight, model[0].weight)
    augmented = TRAIN_TRANSFORM(inputs[0])
    assert augmented.shape == inputs[0].shape and torch.isfinite(augmented).all()
    print('PASS: degradation, classical enhancers, split/ROI, metrics, pairing/bootstrap, cache roundtrip and CNN backward')


if __name__ == '__main__':
    main()
