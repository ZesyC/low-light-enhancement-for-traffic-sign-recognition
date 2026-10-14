"""Frozen clean classifier: degradation ablation and demo assets on Kaggle GPU."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def degradation_variants(medium):
    return {'gamma_only': {**medium, 'poisson_peak': None, 'gaussian_sigma': 0},
            'gamma_gaussian': {**medium, 'poisson_peak': None},
            'gamma_poisson': {**medium, 'gaussian_sigma': 0},
            'gamma_both': dict(medium)}


def main():
    print('EVIDENCE: frozen clean checkpoints; no training or tuning', flush=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    inputs = Path('/kaggle/input')
    sources = [p.parent for p in inputs.rglob('snapshot.json')
               if (p.parent / 'data/manifests/gtsrb_samples.csv').is_file()]
    overlays = list(inputs.rglob('lowlight_snapshot.json'))
    assert len(sources) == len(overlays) == 1
    source, overlay = sources[0], overlays[0].parent
    root = Path('/kaggle/working/gtsrb-evidence')
    root.mkdir()
    for name in ['src', 'configs']:
        shutil.copytree(overlay / name, root / name)
    shutil.copytree(source / 'external', root / 'external')
    (root / 'data').mkdir()
    for name in ['processed', 'manifests']:
        (root / 'data' / name).symlink_to(source / 'data' / name, target_is_directory=True)
    cache = Path('/tmp/gtsrb-evidence-cache')
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
    from PIL import Image
    from src.common import config_id, digest, environment, read_csv, read_image, write_csv, write_json
    from src.data import subset_rows
    from src.degradation import degrade
    from src.experiment import cache_images, load_cnn, predict, source_code_hash
    from src.metrics import classification

    assert torch.cuda.is_available(), 'GPU required'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    snapshot = json.loads(overlays[0].read_text())
    manifest = root / 'data/manifests/gtsrb_samples.csv'
    assert digest(manifest) == snapshot['manifest_sha256']
    assert source_code_hash() == snapshot['source_code_hash']
    all_rows = read_csv(manifest)
    rows = subset_rows(all_rows, 'test', 10, 42)
    assert len(rows) == 430
    selection = json.loads((overlay / 'resnet18_A/validation/selection.json').read_text())
    config = deepcopy(selection['config'])
    config['enhancers'].update(selection['enhancers'])
    models = {}
    for seed in [11, 22, 33]:
        path = overlay / 'resnet18_A/train' / f'cnn_{seed}.pt'
        assert digest(path) == snapshot['baseline_checkpoint_sha256'][path.name]
        models[seed] = load_cnn(path, digest(manifest), device, config['cnn'])[0]
    output = root / 'outputs/research_evidence'
    variants = degradation_variants(config['conditions']['medium'])
    write_json(output / 'protocol.json', {
        'kind': 'degradation_ablation', 'training': False, 'tuning': False,
        'sample_ids': [r['sample_id'] for r in rows], 'manifest_sha256': digest(manifest),
        'source_code_hash': source_code_hash(), 'variants': variants,
        'methods': selection['methods'], 'enhancers': selection['enhancers'],
        'checkpoint_sha256': snapshot['baseline_checkpoint_sha256'],
        'noise_seeds': config['noise_seeds'], 'train_seeds': [11, 22, 33],
        'noise_rule': 'same medium condition and seed input; turning Poisson off changes subsequent Gaussian RNG draws',
        'role': 'supplementary test analysis; does not select hyperparameters or replace protocol A',
        'environment': {**environment(), 'gpu': torch.cuda.get_device_name(0)}})
    summary = []
    for variant, parameters in variants.items():
        cfg = deepcopy(config)
        cfg['conditions']['medium'] = parameters
        for noise in cfg['noise_seeds']:
            for method in selection['methods']:
                cached = cache_images(root, rows, digest(manifest), cfg, 'medium', noise,
                                      method, cfg['enhancers'][method], device)
                for seed, model in models.items():
                    predictions = predict(root, rows, cached, model, cfg, device)
                    score = classification([r['class_id'] for r in predictions], [r['prediction'] for r in predictions])
                    run = {'variant': variant, 'method': method, 'noise_seed': noise, 'train_seed': seed}
                    run_id = config_id(run)
                    write_csv(output / 'predictions' / f'{run_id}.csv', predictions)
                    summary.append({**run, 'run_id': run_id, 'n_images': len(predictions),
                                    **{k: score[k] for k in ['accuracy', 'macro_f1', 'weighted_f1']},
                                    **{k: float(np.mean([float(r[k]) for r in cached])) for k in ['psnr', 'ssim', 'latency_ms']}})
                    write_csv(output / 'summary.csv', summary)
                    print(f'{len(summary)}/288 {run}: macro_f1={score["macro_f1"]:.4f}', flush=True)
    assert len(summary) == 288 and all(r['n_images'] == 430 for r in summary)
    # First validation ID per class, independent of model predictions and visual appearance.
    validation = subset_rows(all_rows, 'validation', 1, 42)
    examples = [next(r for r in validation if int(r['class_id']) == label) for label in range(43)]
    write_json(output / 'demo_ids.json', [r['sample_id'] for r in examples])
    demo_rows = []
    for condition in ['clean', *config['conditions']]:
        methods = ['clean'] if condition == 'clean' else selection['methods']
        for method in methods:
            params = config['enhancers'].get(method, {})
            cached = cache_images(root, examples, digest(manifest), config, condition, 101, method, params, device)
            predictions = predict(root, examples, cached, models[11], config, device)
            for sample, record, prediction in zip(examples, cached, predictions):
                name = config_id({'sample_id': sample['sample_id'], 'condition': condition, 'method': method}) + '.png'
                path = output / 'demo/images' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.rint(read_image(root / record['path']) * 255).astype(np.uint8)).save(path)
                demo_rows.append({'condition': condition, 'method': method, 'image': 'images/' + name, **prediction})
    write_csv(output / 'demo/predictions.csv', demo_rows)
    archive = shutil.make_archive('/kaggle/working/gtsrb-research-evidence', 'zip', root / 'outputs')
    print('COMPLETE:', archive, flush=True)


if __name__ == '__main__':
    main()
