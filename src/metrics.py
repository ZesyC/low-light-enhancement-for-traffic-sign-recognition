from collections import defaultdict

import numpy as np
from skimage.metrics import structural_similarity
from .common import validate_image


def classification(labels, predictions):
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if labels.shape != predictions.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError('Expected nonempty paired label/prediction vectors')
    if min(labels.min(), predictions.min()) < 0 or max(labels.max(), predictions.max()) >= 43:
        raise ValueError('Class IDs must be 0..42')
    matrix = np.bincount(43 * labels + predictions, minlength=43*43).reshape(43, 43)
    support, predicted = matrix.sum(1), matrix.sum(0)
    f1 = np.divide(2*np.diag(matrix), support+predicted, out=np.zeros(43), where=support+predicted > 0)
    return {'accuracy': float(np.trace(matrix) / labels.size), 'macro_f1': float(f1.mean()),
            'weighted_f1': float(f1 @ support / labels.size), 'f1_per_class': f1.tolist(),
            'support': support.tolist(), 'confusion_matrix': matrix.tolist()}


def image_metrics(reference, output, ssim_config):
    reference, output = validate_image(reference), validate_image(output)
    if reference.shape != output.shape:
        raise ValueError('Reference/output shape mismatch')
    mse = np.mean((reference.astype(np.float64)-output)**2)
    return {'psnr': float('inf') if mse == 0 else float(10*np.log10(1/mse)),
            'ssim': float(structural_similarity(reference, output, **ssim_config))}


def paired_rows(baseline, enhanced):
    key = lambda r: (r['sample_id'], str(r['noise_seed']))
    a, b = {key(r): r for r in baseline}, {key(r): r for r in enhanced}
    if len(a) != len(baseline) or len(b) != len(enhanced) or a.keys() != b.keys():
        raise ValueError('Paired sample IDs/noise seeds mismatch or duplicate')
    pairs = [(a[k], b[k]) for k in sorted(a)]
    if any(x['class_id'] != y['class_id'] or x['group_id'] != y['group_id'] for x, y in pairs):
        raise ValueError('Paired labels/groups mismatch')
    return pairs


def paired_bootstrap(baseline, enhanced, repetitions=1000, seed=42):
    pairs = paired_rows(baseline, enhanced)
    groups = defaultdict(list)
    for i, (a, _) in enumerate(pairs):
        groups[a['group_id']].append(i)
    if repetitions < 1:
        raise ValueError('Bootstrap repetitions must be positive')
    values = list(groups.values())
    labels = np.array([int(a['class_id']) for a, _ in pairs])
    before = np.array([int(a['prediction']) for a, _ in pairs])
    after = np.array([int(b['prediction']) for _, b in pairs])
    noise = np.array([str(a['noise_seed']) for a, _ in pairs])
    seeds = np.unique(noise)

    def delta(indices):
        # Same resampled source groups for every noise realization; average metrics over noise.
        result = []
        for ns in seeds:
            idx = indices[noise[indices] == ns]
            a, b = classification(labels[idx], before[idx]), classification(labels[idx], after[idx])
            result.append([100*(b[k]-a[k]) for k in ['accuracy', 'macro_f1']])
        return np.mean(result, axis=0)

    rng = np.random.default_rng(seed)
    draws = [delta(np.concatenate([values[i] for i in rng.integers(len(values), size=len(values))]))
             for _ in range(repetitions)]
    intervals = np.percentile(draws, [2.5, 97.5], axis=0)
    observed = delta(np.arange(len(pairs)))
    return {'n_groups': len(groups), 'repetitions': repetitions, 'unit': 'percentage points',
            'conditional_on_checkpoint': True, 'grouping': 'group_id; test falls back to source_id',
            **{k: {'delta': float(observed[i]), 'ci95': intervals[:, i].tolist()}
               for i, k in enumerate(['accuracy', 'macro_f1'])}}
