"""Kaggle entry point: use the private, audited dataset snapshot."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def select_configuration(root, config, device):
    from src.common import write_json
    from src.training import train

    candidates = []
    for learning_rate in [config['cnn']['learning_rate'], 3e-5, 3e-4]:
        candidate = deepcopy(config)
        candidate['cnn']['learning_rate'] = learning_rate
        folder = root / 'outputs' / f'resnet18_lr_{learning_rate:g}' / 'train'
        train(root, candidate, 11, device, folder)
        score = json.loads((folder / 'validation_11.json').read_text())
        candidates.append((score['macro_f1'], candidate, folder))
        if len(candidates) == 1 and all(score[k] > 0.9 for k in ['accuracy', 'macro_f1', 'weighted_f1']):
            break
    _, selected, folder = max(candidates, key=lambda item: item[0])
    output = root / 'outputs/resnet18_A'
    (output / 'train').mkdir(parents=True, exist_ok=True)
    for name in ['cnn_11.pt', 'train_11.json', 'validation_11.json', 'learning_11.csv']:
        destination = output / 'train' / name
        if destination.exists():
            raise FileExistsError(f'Refusing to replace {destination}')
        shutil.copy2(folder / name, destination)
    write_json(output / 'selected_config.json', selected)
    write_json(output / 'lr_selection.json', {
        'split': 'validation', 'seed': 11, 'criterion': 'macro_f1',
        'trials': [{'learning_rate': cfg['cnn']['learning_rate'], 'macro_f1': f1,
                    'directory': str(path)} for f1, cfg, path in candidates],
        'chosen_learning_rate': selected['cnn']['learning_rate']})
    return selected


def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    inputs = list(Path('/kaggle/input').rglob('snapshot.json'))
    if len(inputs) != 1:
        raise ValueError('Attach exactly one audited dataset containing snapshot.json')
    root = Path('/kaggle/working/gtsrb-resnet18')
    # Kaggle expands the uploaded tar.gz into dataset files.
    shutil.copytree(inputs[0].parent, root)
    os.chdir(root)
    sys.path.insert(0, str(root))
    # Keep Kaggle's CUDA-compatible torch/torchvision; install only missing helper packages.
    missing = [package for module, package in [('einops', 'einops'), ('yaml', 'PyYAML'),
                                               ('pytorch_msssim', 'pytorch-msssim==1.0.0')]
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', *missing], check=True)
    import numpy as np
    import torch
    from src.common import digest, environment, read_csv, write_json
    from src.data import subset_rows
    from src.experiment import evaluate, freeze, tune, source_code_hash
    from src.reporting import report
    from src.training import train

    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required; refusing a long CPU training run')
    torch.set_num_threads(2)
    torch.hub.set_dir(str(root / '.torch'))
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    config = json.loads(Path('configs/resnet18.json').read_text())
    snapshot = json.loads(Path('snapshot.json').read_text())
    assert source_code_hash() == snapshot['source_code_hash'], 'Source snapshot changed'
    for name, expected in snapshot['sha256'].items():
        assert digest(name) == expected, f'Snapshot file changed: {name}'
    rows = read_csv('data/manifests/gtsrb_samples.csv')
    declared = json.loads(Path('outputs/test_subset_plan.json').read_text())
    assert declared['sample_ids'] == [r['sample_id'] for r in subset_rows(rows, 'test', 10, 42)]
    assert all((root / r['crop_path']).is_file() for r in rows)
    write_json('outputs/kaggle_environment.json', {
        **environment(), 'gpu': torch.cuda.get_device_name(0),
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'snapshot_sha256': digest(inputs[0]), 'source_code_hash': source_code_hash()})
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    for module in ['checks.check_pipeline', 'checks.check_resnet18']:
        subprocess.run([sys.executable, '-m', module], check=True)
    subprocess.run([sys.executable, '-m', 'scripts.run', '--device', 'cuda', 'smoke',
                    '--count', '1', '--output', 'outputs/kaggle_smoke'], check=True)
    selected = select_configuration(root, config, device)
    output = root / 'outputs/resnet18_A'
    checkpoints = [train(root, selected, seed, device, output / 'train') for seed in [11, 22, 33]]
    scores = [json.loads((output / 'train' / f'validation_{seed}.json').read_text()) for seed in [11, 22, 33]]
    write_json(output / 'validation_summary.json', {
        'split': 'validation', 'n_images': 7829,
        'metrics': {key: {'mean': float(np.mean([s[key] for s in scores])),
                          'std': float(np.std([s[key] for s in scores], ddof=1))}
                    for key in ['accuracy', 'macro_f1', 'weighted_f1']},
        'per_seed': {str(seed): score for seed, score in zip([11, 22, 33], scores)}})
    tune(root, selected, checkpoints, device, output / 'validation', per_class=30)
    freeze(root, output / 'validation/selection.json', output / 'protocol.json', per_class=10, subset_seed=42)
    result = evaluate(root, output / 'protocol.json', output / 'test_subset', device)
    assert len(result) == 219 and all(r['n_images'] == 430 for r in result)
    print(report(output / 'test_subset', root), flush=True)


if __name__ == '__main__':
    main()
