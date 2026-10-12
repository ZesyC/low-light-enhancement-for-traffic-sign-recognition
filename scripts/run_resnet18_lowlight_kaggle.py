"""GPU pilot, then mixed-lowlight training and paired subset comparison on Kaggle."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    inputs = Path('/kaggle/input')
    sources = [p.parent for p in inputs.rglob('snapshot.json')
               if (p.parent / 'data/manifests/gtsrb_samples.csv').is_file()]
    overlays = list(inputs.rglob('lowlight_snapshot.json'))
    if len(sources) != 1 or len(overlays) != 1:
        raise ValueError('Attach exactly one audited source dataset and one lowlight code snapshot')
    source, overlay = sources[0], overlays[0].parent
    root = Path('/kaggle/working/gtsrb-lowlight')
    root.mkdir()
    for name in ['src', 'configs', 'checks', 'scripts']:
        shutil.copytree(overlay / name, root / name)
    shutil.copytree(source / 'external', root / 'external')
    shutil.copytree(source / '.torch', root / '.torch')
    (root / 'data').mkdir()
    for name in ['processed', 'manifests']:
        (root / 'data' / name).symlink_to(source / 'data' / name, target_is_directory=True)
    cache = Path('/tmp/resnet18-lowlight-cache')
    cache.mkdir()
    (root / 'data/cache').symlink_to(cache, target_is_directory=True)
    os.chdir(root)
    sys.path.insert(0, str(root))
    missing = [p for m, p in [('einops', 'einops'), ('yaml', 'PyYAML'),
                              ('pytorch_msssim', 'pytorch-msssim==1.0.0')]
               if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
    import numpy as np
    import torch
    from src.common import digest, environment, read_csv, write_json, write_csv
    from src.experiment import evaluate, freeze, source_code_hash, tune
    from src.metrics import paired_rows
    from src.reporting import report
    from src.training import train
    from scripts.run_resnet18_full_test_kaggle import copy_previous_results
    from scripts.verify_results import verify

    assert torch.cuda.is_available(), 'GPU required; refusing CPU training'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(root / '.torch'))
    device = torch.device('cuda:0')
    snapshot = json.loads(overlays[0].read_text())
    assert source_code_hash() == snapshot['source_code_hash']
    for name, expected in snapshot['sha256'].items():
        assert digest(root / name) == expected, f'Snapshot mismatch: {name}'
    assert digest(root / 'data/manifests/gtsrb_samples.csv') == snapshot['manifest_sha256']
    assert shutil.disk_usage(cache).free > 20 * 2**30, 'Insufficient temporary cache disk'
    config = json.loads((root / 'configs/resnet18_lowlight.json').read_text())
    clean = root / 'outputs/resnet18_clean_replay'
    copy_previous_results(overlay, clean)
    # Preserve the original selection; replay uses fresh validation/protocol under the new source hash.
    original = json.loads((clean / 'validation/selection.json').read_text())
    baseline_config = original['config']
    checkpoints = [clean / 'train' / f'cnn_{s}.pt' for s in [11, 22, 33]]
    for p in checkpoints:
        assert digest(p) == next(h for path, h in original['checkpoints'].items() if Path(path).name == p.name)
    baseline_summary = json.loads((clean / 'validation_summary.json').read_text())
    assert abs(config['training']['clean_min_macro_f1'] -
               (baseline_summary['metrics']['macro_f1']['mean'] - 0.01)) < 1e-12
    write_json(root / 'outputs/environment.json', {**environment(), 'gpu': torch.cuda.get_device_name(0)})
    print('GPU:', torch.cuda.get_device_name(0), 'MIXED LOWLIGHT: separate experiment, 50% clean', flush=True)
    for module in ['checks.check_pipeline', 'checks.check_resnet18', 'checks.check_lowlight']:
        subprocess.run([sys.executable, '-m', module], check=True)
    pilot = root / 'outputs/resnet18_lowlight_pilot'
    pilot_config = deepcopy(config)
    pilot_config['training']['clean_min_macro_f1'] = 0
    checkpoint = train(root, pilot_config, 11, device, pilot / 'train', per_class=10, epochs=2)
    tune(root, pilot_config, [checkpoint], device, pilot / 'validation', per_class=2, methods=['identity', 'gamma'])
    freeze(root, pilot / 'validation/selection.json', pilot / 'protocol.json', per_class=2, pilot=True)
    results = evaluate(root, pilot / 'protocol.json', pilot / 'test', device)
    assert len(results) == 19 and all(r['n_images'] == 86 for r in results)
    report(pilot / 'test', root)
    verify(root, pilot / 'test')
    print('PILOT COMPLETE: 2 GPU epochs, 19 runs; pilot test never used to select hyperparameters', flush=True)

    destination = root / 'outputs/resnet18_lowlight'
    trained = [train(root, config, seed, device, destination / 'train') for seed in [11, 22, 33]]
    scores = {str(s): json.loads((destination / 'train' / f'validation_{s}.json').read_text()) for s in [11, 22, 33]}
    write_json(destination / 'validation_summary.json', {
        'regime': 'mixed_lowlight', 'clean_min_macro_f1': config['training']['clean_min_macro_f1'],
        'per_seed': scores, 'selection': 'equal mean Macro-F1 on clean/mild/medium/severe; clean guardrail required'})
    print('TRAIN COMPLETE: all 3 seeds passed clean validation guardrail', flush=True)
    tune(root, config, trained, device, destination / 'validation', per_class=30)
    freeze(root, destination / 'validation/selection.json', destination / 'protocol.json', per_class=10, subset_seed=42)
    mixed_results = evaluate(root, destination / 'protocol.json', destination / 'test_subset', device)
    assert len(mixed_results) == 219
    report(destination / 'test_subset', root)
    verify(root, destination / 'test_subset')

    tune(root, baseline_config, checkpoints, device, clean / 'validation_replay', per_class=30)
    freeze(root, clean / 'validation_replay/selection.json', clean / 'protocol_replay.json', per_class=10, subset_seed=42)
    clean_results = evaluate(root, clean / 'protocol_replay.json', clean / 'test_subset', device)
    assert len(clean_results) == 219
    report(clean / 'test_subset', root)
    verify(root, clean / 'test_subset')
    baseline_rows = {(r['condition'], r['method'], r['train_seed'], r['noise_seed']): r for r in clean_results}
    differences = []
    for row in mixed_results:
        key = tuple(row[k] for k in ['condition', 'method', 'train_seed', 'noise_seed'])
        before = baseline_rows[key]
        paired_rows(read_csv(clean / 'test_subset/predictions' / f"{before['run_id']}.csv"),
                    read_csv(destination / 'test_subset/predictions' / f"{row['run_id']}.csv"))
        differences.append({k: row[k] for k in ['condition', 'method', 'train_seed', 'noise_seed']} |
                           {'same_enhancer_parameters': row['config_id'] == before['config_id'],
                            **{f'delta_{k}_pp': 100 * (row[k] - before[k]) for k in ['accuracy', 'macro_f1', 'weighted_f1']}})
    write_csv(destination / 'paired_deltas.csv', differences)
    # Each regime tunes its enhancer on validation; identity isolates the training change.
    aggregate = []
    for condition, method in sorted({(r['condition'], r['method']) for r in differences}):
        entry = {'condition': condition, 'method': method}
        for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
            values = [np.mean([r[f'delta_{metric}_pp'] for r in differences
                               if (r['condition'], r['method'], r['train_seed']) == (condition, method, s)])
                      for s in [11, 22, 33]]
            entry.update({f'delta_{metric}_pp_mean': float(np.mean(values)),
                          f'delta_{metric}_pp_std': float(np.std(values, ddof=1))})
        aggregate.append(entry)
    write_csv(destination / 'paired_aggregate.csv', aggregate)
    write_json(destination / 'comparison_definition.json', {
        'images': 430, 'paired_by': ['sample_id', 'condition', 'noise_seed', 'train_seed'],
        'identity': 'isolates clean vs mixed training regime',
        'enhanced': 'each regime independently tunes enhancement parameters on validation; includes that change',
        'baseline_checkpoint_hashes': {p.name: digest(p) for p in checkpoints},
        'pilot_excluded': True})
    archive = shutil.make_archive('/kaggle/working/resnet18-lowlight-results', 'zip', root / 'outputs')
    print(f'COMPLETE: {archive}', flush=True)


if __name__ == '__main__':
    main()
