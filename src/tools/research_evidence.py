"""Audit raw prediction coverage before producing error analysis or an evidence ZIP."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import zipfile

import numpy as np

from src.common import config_id, digest
from src.metrics import classification, paired_rows


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audited_runs(folder):
    folder = Path(folder)
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    states = json.loads((folder / 'runs.json').read_text(encoding='utf-8'))
    expected = {config_id(r): r for r in protocol['runs']}
    if len(expected) != len(protocol['runs']):
        raise ValueError('Duplicate frozen runs')
    if len(states) != len(expected) or {config_id({k: s[k] for k in ['condition', 'method', 'noise_seed', 'train_seed']}) for s in states} != set(expected) or any(s['status'] != 'completed' for s in states):
        raise ValueError('Run states are incomplete or differ from the frozen protocol')
    summaries = read_csv(folder / 'summary.csv')
    if len(summaries) != len(expected) or {r['run_id'] for r in summaries} != set(expected):
        raise ValueError('Summary does not cover the frozen run matrix')
    ids = set(protocol['sample_ids'])
    if not ids or len(ids) != len(protocol['sample_ids']):
        raise ValueError('Invalid sample IDs in protocol')
    reference_labels = None
    loaded = []
    for summary in summaries:
        run = expected[summary['run_id']]
        if any(str(summary[k]) != str(v) for k, v in run.items()):
            raise ValueError('Summary metadata does not match frozen run')
        if int(summary['n_images']) != len(ids):
            raise ValueError('Incorrect sample count')
        rows = read_csv(folder / 'predictions' / f"{summary['run_id']}.csv")
        if len(rows) != len(ids) or {r['sample_id'] for r in rows} != ids:
            raise ValueError('Missing or duplicate predictions')
        for r in rows:
            r['class_id'], r['prediction'] = int(r['class_id']), int(r['prediction'])
            if r['noise_seed'] != str(run['noise_seed']):
                raise ValueError('Prediction noise seed mismatch')
        labels = {r['sample_id']: (r['class_id'], r['group_id']) for r in rows}
        if reference_labels is None:
            reference_labels = labels
        elif labels != reference_labels:
            raise ValueError('Label/group mismatch between runs')
        score = classification([r['class_id'] for r in rows], [r['prediction'] for r in rows])
        for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
            if not math.isclose(score[metric], float(summary[metric]), rel_tol=0, abs_tol=1e-8):
                raise ValueError(f'Metric mismatch: {summary["run_id"]} {metric}')
        loaded.append((summary, rows, score))
    if not protocol.get('pilot', True):
        from src.enhancement import METHODS
        methods=METHODS
        if protocol.get('study')=='denoise_order_extension':
            methods=protocol['selection']['methods']
            if (len(methods)!=len(set(methods)) or set(methods)!=set(protocol['registry'])
                or not {'identity','gamma','clahe'}<=set(methods)
                or protocol['selection']['chosen_method'] not in methods):
                raise ValueError('Invalid extension method registry or controls')
            if any(protocol['registry'][m]!=protocol['selection']['registry'][m] for m in methods):
                raise ValueError('Extension pipeline differs from validation selection')
            intended=(protocol['selection']['fulltest_methods'] if protocol['subset_id']=='official_test'
                      else list(protocol['selection']['registry']))
            if set(methods)!=set(intended):raise ValueError('Extension test methods differ from selection')
        elif protocol.get('study') is not None:
            raise ValueError('Unknown study')
        cfg = protocol['config']
        if set(cfg['conditions']) != {'mild', 'medium', 'severe'} or set(cfg['noise_seeds']) != {101, 202, 303} or {c['seed'] for c in protocol['checkpoints']} != {11, 22, 33}:
            raise ValueError('Final experiment requires all three levels and seeds')
        required = {
            config_id(dict(condition=c, method=m, noise_seed=n, train_seed=t))
            for c in cfg['conditions'] for m in methods for n in cfg['noise_seeds'] for t in [11, 22, 33]
        } | {config_id(dict(condition='clean', method='clean', noise_seed='', train_seed=t)) for t in [11, 22, 33]}
        if set(expected) != required or {r['class_id'] for r in loaded[0][1]} != set(range(43)):
            raise ValueError('Final experiment matrix/classes incomplete')
        if protocol['subset_id'] == 'official_test' and len(ids) != 12630:
            raise ValueError('Official full-test must contain 12,630 samples')
    return protocol, loaded


def analyze(folder, output):
    protocol, runs = audited_runs(folder)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new error-analysis output directory')
    output.mkdir(parents=True)
    baselines = {(r['condition'], r['train_seed'], r['noise_seed']): p for r,p,_ in runs if r['method'] == 'identity'}
    per_class, transitions, confusions, cases = [], [], [], []
    for run, predictions, score in runs:
        meta = {k: run[k] for k in ['run_id', 'condition', 'method', 'train_seed', 'noise_seed']}
        matrix = np.array(score['confusion_matrix'])
        np.savetxt(output / f"confusion_{run['run_id']}.csv", matrix, delimiter=',', fmt='%d')
        normalized = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(1, keepdims=True) != 0)
        np.savetxt(output / f"confusion_normalized_{run['run_id']}.csv", normalized, delimiter=',', fmt='%.8f')
        for c in range(43):
            per_class.append({**meta, 'class_id': c, 'support': score['support'][c], 'f1': score['f1_per_class'][c]})
            for p in range(43):
                if p != c and matrix[c,p]:
                    confusions.append({**meta, 'true_class': c, 'predicted_class': p, 'count': int(matrix[c,p]), 'row_fraction': float(normalized[c,p])})
        if run['method'] == 'clean':
            continue
        key = (run['condition'], run['train_seed'], run['noise_seed'])
        if key not in baselines:
            raise ValueError('Missing matched identity baseline')
        counts = Counter()
        for before, after in paired_rows(baselines[key], predictions):
            old = before['prediction'] == before['class_id']
            new = after['prediction'] == after['class_id']
            category = 'helped' if new and not old else 'harmed' if old and not new else 'still_correct' if new else 'still_wrong'
            counts[category] += 1
            cases.append({**meta, 'sample_id': after['sample_id'], 'class_id': after['class_id'], 'category': category,
                          'before': before['prediction'], 'after': after['prediction'],
                          'before_path': before['output_path'], 'after_path': after['output_path']})
        transitions.append({**meta, **{k: counts[k] for k in ['helped', 'harmed', 'still_correct', 'still_wrong']},
                            'net_correct': counts['helped']-counts['harmed']})
    write_csv(output / 'per_class_runs.csv', per_class)
    write_csv(output / 'transitions.csv', transitions)
    write_csv(output / 'cases.csv', cases)
    write_csv(output / 'confusion_pairs.csv', confusions, ['run_id','condition','method','train_seed','noise_seed','true_class','predicted_class','count','row_fraction'])
    # Average F1 within checkpoint over noise, then across checkpoints. Never pool seeds.
    groups = defaultdict(list)
    for r in per_class:
        groups[r['condition'],r['method'],r['class_id'],r['train_seed']].append(r['f1'])
    across_seeds = defaultdict(list)
    for (c,m,label,_), values in groups.items():
        across_seeds[c,m,label].append(float(np.mean(values)))
    baseline = {(c,label): np.mean(v) for (c,m,label),v in across_seeds.items() if m == 'identity'}
    aggregates = []
    for (c,m,label),values in sorted(across_seeds.items()):
        mean = float(np.mean(values))
        aggregates.append(dict(condition=c,method=m,class_id=label,f1_mean=mean,
                               f1_sd=float(np.std(values, ddof=1)) if len(values)>1 else '',
                               n_train_seeds=len(values), delta_f1_pp=100*(mean-baseline[c,label]) if (c,label) in baseline else ''))
    write_csv(output / 'per_class_aggregate.csv', aggregates)
    # IDs, method and seeds are retained; repeated source images are not independent examples.
    examples = []
    selected = protocol['selection']['chosen_method']
    for method in sorted({selected, 'clahe'}):
        for category in ['helped','harmed','still_wrong']:
            subset = sorted((r for r in cases if r['method']==method and r['category']==category), key=lambda r: (r['sample_id'],r['condition'],r['train_seed'],r['noise_seed']))
            seen = set()
            for r in subset:
                if r['sample_id'] not in seen:
                    examples.append(r)
                    seen.add(r['sample_id'])
                    if len(seen) == 5:
                        break
    write_csv(output / 'examples.csv', examples, list(cases[0]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for run, _, score in runs:
        if run['method'] not in {'identity', selected, 'clahe'} or str(run['train_seed']) != str(protocol['checkpoints'][0]['seed']) or str(run['noise_seed']) != str(protocol['config']['noise_seeds'][0]):
            continue
        matrix = np.array(score['confusion_matrix'], dtype=float)
        matrix = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(1, keepdims=True)!=0)
        fig, ax = plt.subplots(figsize=(9,8))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        ax.set(xlabel='Predicted class', ylabel='True class', title=f"{run['method']} / {run['condition']} / train {run['train_seed']} / noise {run['noise_seed']}")
        ax.set_xticks(range(0,43,2)); ax.set_yticks(range(0,43,2))
        fig.colorbar(im, ax=ax, label='Fraction within true class')
        fig.tight_layout(); fig.savefig(output / f"confusion_{run['run_id']}.png", dpi=150); plt.close(fig)
    audit = {'status':'raw_predictions_recomputed', 'runs':len(runs), 'images_per_run':len(protocol['sample_ids']),
             'scope':protocol['subset_id'], 'pilot':protocol.get('pilot',True),
             'protocol_sha256':digest(Path(folder)/'protocol.json'),
             'source_hashes':{str(p.relative_to(folder)):digest(p) for p in sorted(Path(folder).rglob('*.csv')) if p.parent.name=='predictions' or p.name=='summary.csv'},
             'aggregation':'Average noise within checkpoint, then mean and sample SD across checkpoints; confusion counts remain per run'}
    (output / 'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return output


def bundle(folder, output):
    audited_runs(folder)
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise FileExistsError('Use a new ZIP path')
    required = ['protocol.json','runs.json','summary.csv']
    paths = [folder / p for p in required]
    paths += sorted((folder/'predictions').glob('*.csv'))
    for name in ['classes','train','validation','error_analysis','error_analysis_prod']:
        paths += sorted(p for p in (folder/name).rglob('*') if p.is_file())
    for name in ['environment.json','classifier_latency.json','aggregate.csv','per_class.csv','bootstrap.json','cases.csv']:
        if (folder/name).is_file():
            paths.append(folder/name)
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest = {str(p.relative_to(folder)).replace('\\','/'):digest(p) for p in paths}
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p,p.relative_to(folder).as_posix())
        archive.writestr('SHA256.json',json.dumps(manifest,indent=2))
        archive.writestr('SCOPE.txt','Raw prediction evidence. Does not include original images, cached enhancer arrays, external weights or classifier checkpoints. Preserve those separately.\n')
    return output
