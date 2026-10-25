"""Stream-audit an exported results ZIP before using its tables in the report."""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

from src.common import config_id, digest, write_csv, write_json
from src.metrics import classification
from .result_archive import aggregate


def csv_rows(archive, name):
    with archive.open(name) as binary:
        yield from csv.DictReader(io.TextIOWrapper(binary, encoding='utf-8-sig', newline=''))


def audit_scope(archive, protocol_name, summary_name, predictions_name, matrices_name=None):
    protocol = json.loads(archive.read(protocol_name))
    expected = {config_id(row): row for row in protocol['runs']}
    if len(expected) != len(protocol['runs']):
        raise ValueError('Duplicate frozen run')
    if protocol.get('pilot', True):
        raise ValueError('Pilot runs cannot supply final report evidence')
    config = protocol['config']
    if set(config['conditions']) != {'mild', 'medium', 'severe'} or set(config['noise_seeds']) != {101, 202, 303}:
        raise ValueError('Incomplete degradation conditions or noise seeds')
    methods = protocol['selection']['methods'] if protocol.get('study') == 'denoise_order_extension' else [
        'identity', 'gamma', 'he', 'clahe', 'retinex', 'zero_dce', 'retinexformer', 'diffusion']
    planned = {config_id(dict(condition=c, method=m, noise_seed=n, train_seed=t))
        for c in config['conditions'] for m in methods for n in config['noise_seeds'] for t in [11, 22, 33]}
    planned |= {config_id(dict(condition='clean', method='clean', noise_seed='', train_seed=t)) for t in [11, 22, 33]}
    if set(expected) != planned:
        raise ValueError('Frozen matrix omits a required method, condition or seed')
    summaries = list(csv_rows(archive, summary_name))
    by_run = {row['run_id']: row for row in summaries}
    if len(by_run) != len(summaries) or set(by_run) != set(expected):
        raise ValueError('Summary differs from frozen run matrix')
    ids = set(protocol['sample_ids'])
    if not ids or len(ids) != len(protocol['sample_ids']):
        raise ValueError('Invalid frozen sample IDs')
    if protocol['subset_id'] == 'official_test' and len(ids) != 12630:
        raise ValueError('Full-test requires 12,630 samples')
    matrices = {}
    if matrices_name:
        with np.load(io.BytesIO(archive.read(matrices_name)), allow_pickle=False) as saved:
            matrices = dict(zip(saved['run_ids'], saved['matrices']))
        if set(matrices) != set(expected):
            raise ValueError('Confusion matrix coverage differs')
    current = None
    labels, predictions, samples = [], [], set()
    reference, checked, count = {}, set(), 0

    def flush():
        if current is None:
            return
        if current in checked or samples != ids:
            raise ValueError('Duplicate or incomplete prediction run')
        summary, run = by_run[current], expected[current]
        if int(summary['n_images']) != len(ids):
            raise ValueError('Wrong sample count in summary')
        if any(str(summary[key]) != str(value) for key, value in run.items()):
            raise ValueError('Summary run metadata differs')
        score = classification(labels, predictions)
        for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
            if not math.isclose(score[metric], float(summary[metric]), rel_tol=0, abs_tol=1e-8):
                raise ValueError('Metric mismatch: ' + metric)
        if matrices and not np.array_equal(np.asarray(score['confusion_matrix']), matrices[current]):
            raise ValueError('Confusion matrix mismatch')
        checked.add(current)

    for row in csv_rows(archive, predictions_name):
        rid = row['run_id']
        if rid not in expected:
            raise ValueError('Unplanned prediction run')
        if rid != current:
            flush()
            current = rid
            labels, predictions, samples = [], [], set()
        if any(row[key] != str(expected[rid][key]) for key in ['condition', 'method', 'noise_seed', 'train_seed']):
            raise ValueError('Prediction run metadata differs')
        sample = row['sample_id']
        if sample not in ids or sample in samples:
            raise ValueError('Unexpected or duplicate sample ID')
        truth = (int(row['class_id']), row['group_id'])
        if sample in reference and reference[sample] != truth:
            raise ValueError('Ground truth or group changed across runs')
        reference[sample] = truth
        samples.add(sample)
        labels.append(truth[0])
        predictions.append(int(row['prediction']))
        count += 1
    flush()
    if checked != set(expected) or {value[0] for value in reference.values()} != set(range(43)):
        raise ValueError('Incomplete runs or class coverage')
    points = aggregate(summaries)
    return protocol, summaries, points, {
        'runs': len(checked), 'images_per_run': len(ids), 'prediction_rows': count,
        'confusion_matrices_verified': len(matrices), 'metrics_tolerance': 1e-8,
    }


def correlations(points):
    grouped = defaultdict(list)
    for row in points:
        if row['condition'] != 'clean':
            grouped[row['condition']].append(row)

    def ranks(values):
        return np.array([1 + sum(w < v for w in values) + (sum(w == v for w in values) - 1) / 2 for v in values])

    result = []
    for condition, rows in sorted(grouped.items()):
        for quality in ['psnr_mean', 'ssim_mean']:
            for metric in ['accuracy', 'macro_f1']:
                x = np.array([row[quality + '_mean'] for row in rows])
                y = np.array([row[metric + '_mean'] for row in rows])
                result.append(dict(condition=condition, quality=quality, metric=metric, n=len(rows),
                    pearson=float(np.corrcoef(x, y)[0, 1]),
                    spearman=float(np.corrcoef(ranks(x), ranks(y))[0, 1])))
    return result


def prepare(archive_path, output, require_extension=False):
    archive_path, output = Path(archive_path), Path(output)
    if output.exists():
        raise FileExistsError('Use a new evidence directory')
    scopes = {
        'baseline_fulltest': ('04_protocol/protocol.json', '01_summary/runs.csv',
            '03_predictions/test.csv', '02_analysis/confusion_matrices.npz'),
        'baseline_subset': ('07_supplemental/subset/protocol.json', '07_supplemental/subset/summary.csv',
            '07_supplemental/subset/predictions.csv', None),
    }
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
            raise ValueError('Duplicate or unsafe archive path')
        manifest = json.loads(archive.read('SHA256.json'))
        if set(manifest) != set(names) - {'SHA256.json'}:
            raise ValueError('Hash manifest coverage differs')
        for name, expected_hash in manifest.items():
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != expected_hash:
                raise ValueError('Checksum mismatch: ' + name)
        for scope in ['subset', 'fulltest']:
            prefix = '08_processing_extension/' + scope + '/'
            if prefix + 'protocol.json' in names:
                scopes['extension_' + scope] = tuple(prefix + n for n in [
                    'protocol.json', 'summary.csv', 'predictions.csv', 'confusion_matrices.npz'])
            elif require_extension:
                raise ValueError('Missing completed extension ' + scope)
        results, records, checkpoints = {}, {}, None
        for scope, paths in scopes.items():
            protocol, summaries, points, record = audit_scope(archive, *paths)
            hashes = {str(row['seed']): row['sha256'] for row in protocol['checkpoints']}
            if checkpoints is None:
                checkpoints = hashes
            if hashes != checkpoints or set(hashes) != {'11', '22', '33'}:
                raise ValueError('Classifier checkpoints differ across scopes')
            if not scope.startswith('extension') and len(protocol['runs']) != 219:
                raise ValueError('Incomplete baseline run matrix')
            results[scope] = (summaries, points)
            records[scope] = record
            print(scope, record, flush=True)
        for seed, sha in checkpoints.items():
            if manifest.get(f'05_training/cnn_{seed}.pt') != sha:
                raise ValueError('Checkpoint bytes differ from protocol')
        # Publish derived tables only after every scope and file has passed.
        output.mkdir(parents=True)
        for scope, (summaries, points) in results.items():
            folder = output / scope
            folder.mkdir()
            write_csv(folder / 'runs.csv', summaries)
            write_csv(folder / 'aggregate.csv', points)
            write_csv(folder / 'correlations.csv', correlations(points))
        for name in names:
            if (name.startswith(('02_analysis/', '04_protocol/validation/', '05_training/', '08_processing_extension/'))
                    and name.endswith(('.json', '.csv')) and '/predictions' not in name):
                path = output / 'source' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(name))
        record = {'archive': str(archive_path.resolve()), 'archive_sha256': digest(archive_path),
            'status': 'all_exported_files_and_raw_test_predictions_verified', 'scopes': records,
            'checkpoint_hashes': checkpoints, 'aggregation': 'Mean noise within checkpoint, then mean and sample SD across checkpoints.'}
        write_json(output / 'audit.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-extension', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output, args.require_extension), indent=2))
