"""Evaluate frozen v3 classifiers on official GTSRB test; never train or tune."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def copy_previous_results(inputs, destination):
    selections = list(inputs.rglob('resnet18_A/validation/selection.json'))
    if selections:
        previous = selections[0].parent.parent
        for name in ['train', 'validation']:
            shutil.copytree(previous / name, destination / name)
        for name in ['selected_config.json', 'validation_summary.json']:
            shutil.copy2(previous / name, destination / name)
    else:
        found = False
        for archive in inputs.rglob('*.zip'):
            with zipfile.ZipFile(archive) as z:
                selected = [n for n in z.namelist() if '/outputs/resnet18_A/' in n and
                            any('/outputs/resnet18_A/' + part in n for part in
                                ['train/', 'validation/', 'selected_config.json', 'validation_summary.json'])]
                if not any(n.endswith('/validation/selection.json') for n in selected):
                    continue
                if found:
                    raise ValueError('Multiple previous result archives')
                for name in selected:
                    relative = Path(name.split('/outputs/resnet18_A/', 1)[1])
                    if '..' in relative.parts or relative.is_absolute():
                        raise ValueError('Unsafe archive path')
                    if not name.endswith('/'):
                        target = destination / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(name) as src, target.open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                found = True
        if not found:
            raise FileNotFoundError('Attach completed v3 kernel output with validation/selection.json')


def main():
    print('FULL TEST: preparing existing v3 checkpoints; no training', flush=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    inputs = Path('/kaggle/input')
    snapshots = [p.parent for p in inputs.rglob('snapshot.json')
                 if (p.parent / 'configs/resnet18.json').is_file()]
    if not snapshots:
        raise FileNotFoundError('Audited source dataset is missing')
    source = snapshots[0]
    root = Path('/kaggle/working/gtsrb-resnet18')
    root.mkdir()
    for name in ['src', 'configs']:
        (root / name).symlink_to(source / name, target_is_directory=True)
    # Copy repository metadata too: Git must inspect a locally owned checkout.
    shutil.copytree(source / 'external', root / 'external')
    for repo in (root / 'external').iterdir():
        if repo.is_dir() and (repo / '.git').exists():
            commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                    check=True, capture_output=True, text=True).stdout.strip()
            print(f'External repository {repo.name}: {commit}', flush=True)
    (root / 'data').mkdir()
    for name in ['processed', 'manifests']:
        (root / 'data' / name).symlink_to(source / 'data' / name, target_is_directory=True)
    cache = Path('/tmp/resnet18-full-cache')
    cache.mkdir()
    (root / 'data/cache').symlink_to(cache, target_is_directory=True)
    destination = root / 'outputs/resnet18_A'
    copy_previous_results(inputs, destination)
    os.chdir(root)
    sys.path.insert(0, str(root))
    missing = [p for m, p in [('einops', 'einops'), ('yaml', 'PyYAML'),
                              ('pytorch_msssim', 'pytorch-msssim==1.0.0')]
               if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
    import torch
    from src.common import digest, read_csv, write_json, write_csv
    from src.experiment import freeze, evaluate, source_code_hash
    from src.reporting import report
    assert torch.cuda.is_available(), 'GPU required'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    selection = json.loads((destination / 'validation/selection.json').read_text())
    assert selection['code_hash'] == source_code_hash(), 'Must use original evaluated source'
    assert all(digest(path) == expected for path, expected in selection['checkpoints'].items())
    cfg = selection['config']['cnn']
    assert (cfg['epochs'], cfg['patience'], cfg['min_delta']) == (20, 5, 0.001)
    # ponytail: retain full float32 cache for replay; shard jobs if temporary disk is insufficient.
    required = 73 * 12630 * (64 * 64 * 3 * 4 + 128)
    free = shutil.disk_usage(cache).free
    print(f'GPU={torch.cuda.get_device_name(0)}; cache free={free/2**30:.1f} GiB; needed~{required/2**30:.1f} GiB', flush=True)
    if free < required * 1.15:
        raise RuntimeError('Insufficient temporary disk for full evaluation cache')
    protocol_path = destination / 'protocol_full.json'
    protocol = freeze(root, destination / 'validation/selection.json', protocol_path)
    assert len(protocol['sample_ids']) == 12630 and len(protocol['runs']) == 219
    output = destination / 'test_full'
    print('FROZEN: 12630 official test images, 219 runs, original checkpoints and enhancement parameters', flush=True)
    results = evaluate(root, protocol_path, output, torch.device('cuda:0'))
    assert len(results) == 219 and all(r['n_images'] == 12630 for r in results)
    print(report(output, root), flush=True)
    import numpy as np
    losses = []
    for row in results:
        predictions = read_csv(output / 'predictions' / f"{row['run_id']}.csv")
        p = np.array([float(r['true_probability']) for r in predictions])
        losses.append({k: row[k] for k in ['run_id', 'condition', 'method', 'train_seed', 'noise_seed']} |
                      {'test_nll': float(-np.log(np.maximum(p, np.finfo(np.float32).tiny)).mean()),
                       'clipped_probabilities': int((p < np.finfo(np.float32).tiny).sum())})
    write_csv(output / 'test_loss.csv', losses)
    write_json(output / 'loss_definition.json', {'test_nll': 'Mean -log(true_probability); float32 tiny floor, clipping count reported.',
                                               'train_validation': 'Cross entropy in train/learning_<seed>.csv; training includes augmentation.'})
    archive = shutil.make_archive('/kaggle/working/resnet18-full-results', 'zip', root / 'outputs')
    print(f'COMPLETE: {archive}', flush=True)


if __name__ == '__main__':
    main()
