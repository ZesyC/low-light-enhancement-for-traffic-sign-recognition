"""Verify downloaded predictions, metrics, checkpoint hashes and dataset coverage."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from src.common import config_id, digest, read_csv, write_json
from src.metrics import classification


def verify(root, output):
    root, output = Path(root), Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    manifest = root / 'data/manifests/gtsrb_samples.csv'
    assert digest(manifest) == protocol['manifest_sha256'], 'Manifest hash mismatch'
    labels = {r['sample_id']: int(r['class_id']) for r in read_csv(manifest) if r['split'] == 'test'}
    expected = set(protocol['sample_ids'])
    assert len(expected) == len(protocol['sample_ids']) and expected <= labels.keys()
    if protocol['subset_id'] == 'official_test':
        assert expected == labels.keys(), 'Official test coverage mismatch'
    states = json.loads((output / 'runs.json').read_text())
    assert len(states) == len(protocol['runs']) and all(r['status'] == 'completed' for r in states)
    summary = read_csv(output / 'summary.csv')
    run_ids = {config_id(r) for r in protocol['runs']}
    assert len(summary) == len(run_ids) and {r['run_id'] for r in summary} == run_ids
    for c in protocol['checkpoints']:
        assert digest(output.parent / 'train' / Path(c['path']).name) == c['sha256']
    grouped = defaultdict(list)
    for row in summary:
        preds = read_csv(output / 'predictions' / f"{row['run_id']}.csv")
        assert len(preds) == int(row['n_images']) == len(expected)
        assert {p['sample_id'] for p in preds} == expected
        assert all(int(p['class_id']) == labels[p['sample_id']] and 0 <= int(p['prediction']) < 43
                   and 0 <= float(p['true_probability']) <= 1 for p in preds)
        score = classification([int(p['class_id']) for p in preds], [int(p['prediction']) for p in preds])
        for key in ['accuracy', 'macro_f1', 'weighted_f1']:
            assert math.isclose(score[key], float(row[key]), abs_tol=1e-12), (row['run_id'], key)
            grouped[row['condition'], row['method'], key, row['train_seed']].append(score[key])
    for row in read_csv(output / 'aggregate.csv'):
        for key in ['accuracy', 'macro_f1', 'weighted_f1']:
            means = [np.mean(values) for (c, m, k, seed), values in grouped.items()
                     if (c, m, k) == (row['condition'], row['method'], key)]
            assert len(means) == int(row['n_training_seeds'])
            assert math.isclose(float(row[key + '_mean']), float(np.mean(means)), abs_tol=1e-12)
            assert math.isclose(float(row[key + '_std']), float(np.std(means, ddof=1)) if len(means) > 1 else 0,
                                abs_tol=1e-12)
    result = {'status': 'verified', 'runs': len(summary), 'images_per_run': len(expected),
              'predictions_checked': len(summary) * len(expected), 'manifest_sha256': digest(manifest),
              'protocol_sha256': digest(output / 'protocol.json'),
              'checks': ['checkpoint hashes', 'test labels and exact ID coverage',
                         'Accuracy/Macro-F1/Weighted-F1 recomputed', 'mean and SD across training seeds']}
    write_json(output / 'verification.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    verify(args.root, args.output)
