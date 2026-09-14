"""Shared float cache, validation search, frozen protocol and paired evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import (config_id, digest, environment, read_csv, read_image, save_array,
                     synchronize, write_csv, write_json)
from .data import subset_rows
from .degradation import degrade
from .enhancement import Enhancer, METHODS, provenance, validation_grid
from .metrics import classification, image_metrics
from .model import make_cnn, image_tensor


def source_code_hash():
    return config_id({p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def manifest_rows(root):
    manifest = Path(root) / 'data/manifests/gtsrb_samples.csv'
    if digest(manifest) != json.loads(manifest.with_suffix('.json').read_text())['sha256']:
        raise ValueError('Manifest has changed')
    audit_path = manifest.parent / 'audit.json'
    if not audit_path.exists() or json.loads(audit_path.read_text())['status'] != 'completed':
        raise ValueError('Data audit is missing or failed; run check-data')
    return read_csv(manifest), digest(manifest)


def cache_images(root, rows, manifest_hash, config, condition, noise_seed, method, params, device):
    """Cache each output once across CNN seeds. Never accept partially complete runs."""
    degradation = {} if condition == 'clean' else config['conditions'][condition]
    metadata = {'manifest_sha256': manifest_hash, 'code_hash': source_code_hash(),
                'condition': condition, 'noise_seed': noise_seed, 'degradation': degradation,
                'enhancer': {'method': 'clean'} if method == 'clean' else provenance(method, params),
                'device': str(device), 'ssim': config['ssim'],
                'sample_ids': [r['sample_id'] for r in rows]}
    folder = Path(root) / 'data/cache' / config_id(metadata)
    index = folder / 'images.csv'
    if index.exists():
        cached = read_csv(index)
        if [r['sample_id'] for r in cached] != metadata['sample_ids']:
            raise ValueError('Cache sample identity mismatch')
        if any(not (Path(root) / r['path']).is_file() or digest(Path(root) / r['path']) != r['sha256'] for r in cached):
            raise ValueError('Cache missing or modified; move this cache folder aside and rerun')
        return cached
    write_json(folder / 'metadata.json', dict(metadata, status='running', environment=environment()))
    enhancer = None if method == 'clean' else Enhancer(method, params, device)
    warmup_ms = 0.0
    if enhancer is not None:
        image = degrade(read_image(Path(root) / rows[0]['crop_path']), **degradation,
                        noise_seed=noise_seed, sample_id=rows[0]['sample_id'], condition=condition)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(2):
            enhancer(image, rows[0]['sample_id'])
        synchronize(device)
        warmup_ms = (time.perf_counter()-start)*1000
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    try:
        for row in rows:
            reference = read_image(Path(root) / row['crop_path'])
            image = reference if method == 'clean' else degrade(reference, **degradation, noise_seed=noise_seed,
                                                               sample_id=row['sample_id'], condition=condition)
            synchronize(device)
            start = time.perf_counter()
            output = reference if enhancer is None else enhancer(image, row['sample_id'])
            synchronize(device)
            latency = 0.0 if enhancer is None else (time.perf_counter()-start)*1000
            if output.shape != reference.shape:
                raise ValueError('Enhancer changed canonical image dimensions')
            path = folder / (config_id(row['sample_id']) + '.npy')
            save_array(path, output)
            records.append({'sample_id': row['sample_id'], 'condition': condition, 'noise_seed': noise_seed,
                            **{k: degradation.get(k, '') for k in ['gamma_dark', 'poisson_peak', 'gaussian_sigma']},
                            'method': method, 'config_id': config_id(params),
                            'path': path.relative_to(root).as_posix(), 'sha256': digest(path),
                            'latency_ms': latency, 'input_mean': float(image.mean()),
                            'output_mean': float(output.mean()),
                            'input_clipped_fraction': float(np.mean((image == 0) | (image == 1))),
                            **image_metrics(reference, output, config['ssim'])})
        write_csv(index, records)
        write_json(folder / 'metadata.json', dict(metadata, status='completed', environment=environment(),
                   warmup_ms=warmup_ms, batch_size=1, resolution=[64, 64], precision='float32',
                   latency_includes='tensor transfer and synchronized inference; excludes file I/O and metrics',
                   peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
    except Exception as error:
        write_json(folder / 'metadata.json', dict(metadata, status='failed', error=str(error)))
        raise
    return records


def load_cnn(checkpoint, manifest_hash, device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved['manifest_sha256'] != manifest_hash:
        raise ValueError('CNN trained on a different manifest')
    model = make_cnn().to(device)
    model.load_state_dict(saved['state_dict'])
    return model.eval(), saved


@torch.inference_mode()
def predict(root, rows, cached, model, config, device):
    if [r['sample_id'] for r in rows] != [r['sample_id'] for r in cached]:
        raise ValueError('Input/output sample IDs do not match')
    predictions = []
    batch_size = config['cnn']['batch_size']
    for start in range(0, len(rows), batch_size):
        batch = cached[start:start+batch_size]
        images = torch.stack([image_tensor(read_image(Path(root) / r['path'])) for r in batch]).to(device)
        probabilities = model(images).softmax(1).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError('Nonfinite CNN predictions')
        for row, output, probabilities_i in zip(rows[start:start+batch_size], batch, probabilities):
            predictions.append({'sample_id': row['sample_id'], 'group_id': row['group_id'],
                                'source_id': row['source_id'], 'class_id': int(row['class_id']),
                                'prediction': int(probabilities_i.argmax()),
                                'true_probability': float(probabilities_i[int(row['class_id'])]),
                                'noise_seed': output['noise_seed'], 'psnr': float(output['psnr']),
                                'ssim': float(output['ssim']), 'latency_ms': float(output['latency_ms']),
                                'output_path': output['path']})
    return predictions


def choose_pipeline(scores, tolerance=0.005):
    if not scores:
        raise ValueError('No validation scores')
    best = max(r['macro_f1'] for r in scores)
    eligible = [r for r in scores if best - r['macro_f1'] < tolerance or r['macro_f1'] == best]
    return min(eligible, key=lambda r: (r['latency_ms'], -r['macro_f1'], r['method']))


def tune(root, config, checkpoints, device, output, per_class=None, methods=METHODS):
    root, output = Path(root), Path(output)
    if (output / 'selection.json').exists():
        raise FileExistsError('Selection already exists; use a new output directory')
    if 'identity' not in methods or len(set(methods)) != len(methods):
        raise ValueError('Validation must include identity and unique methods')
    all_rows, manifest_hash = manifest_rows(root)
    rows = subset_rows(all_rows, 'validation', per_class)
    models = [load_cnn(p, manifest_hash, device) for p in checkpoints]
    seeds = [s['seed'] for _, s in models]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Duplicate CNN training seeds')
    scores, details, selected_params = [], [], {}
    for method in methods:
        candidates = []
        for params in validation_grid(method, config['enhancers'][method]):
            f1s, timings = [], []
            for condition in config['conditions']:
                for noise in config['noise_seeds']:
                    cached = cache_images(root, rows, manifest_hash, config, condition, noise, method, params, device)
                    timings.append(np.mean([float(r['latency_ms']) for r in cached]))
                    for model, saved in models:
                        predictions = predict(root, rows, cached, model, config, device)
                        score = classification([r['class_id'] for r in predictions], [r['prediction'] for r in predictions])
                        f1s.append(score['macro_f1'])
                        run = {'method': method, 'config_id': config_id(params), 'condition': condition,
                               'noise_seed': noise, 'train_seed': saved['seed'], 'macro_f1': score['macro_f1'],
                               'accuracy': score['accuracy'], 'n_images': len(rows)}
                        details.append(run)
                        write_csv(output / 'predictions' / f'{config_id(run)}.csv', predictions)
            candidates.append({'method': method, 'params': params, 'macro_f1': float(np.mean(f1s)),
                               'latency_ms': float(np.mean(timings))})
            print(f"validation {method} {params}: {np.mean(f1s):.5f}", flush=True)
        winner = max(candidates, key=lambda r: (r['macro_f1'], -r['latency_ms']))
        selected_params[method] = winner['params']
        scores.append(winner)
        write_csv(output / 'validation_runs.csv', details)
        write_json(output / f'grid_{method}.json', candidates)
    chosen = choose_pipeline(scores, config['selection_tolerance'])
    result = {'status': 'completed', 'split': 'validation', 'manifest_sha256': manifest_hash,
              'config': config, 'code_hash': source_code_hash(), 'environment': environment(),
              'checkpoints': {str(Path(p).resolve()): digest(p) for p in checkpoints},
              'sample_ids': [r['sample_id'] for r in rows], 'methods': list(methods),
              'enhancer_provenance': {m: provenance(m, selected_params[m]) for m in methods},
              'enhancers': selected_params, 'chosen_method': chosen['method'], 'scores': scores,
              'validation_subset': per_class is not None,
              'pilot': any(s['pilot'] for _, s in models) or set(methods) != set(METHODS),
              'averaging': 'equal weight over conditions, noise seeds, then CNN seeds'}
    write_json(output / 'selection.json', result)
    return result


def freeze(root, selection_path, output, per_class=None, subset_seed=42, pilot=False):
    output, root = Path(output), Path(root)
    if output.exists():
        raise FileExistsError('Protocol already frozen; use a new filename')
    selection = json.loads(Path(selection_path).read_text())
    rows, manifest_hash = manifest_rows(root)
    config = deepcopy(selection['config'])
    if selection['manifest_sha256'] != manifest_hash or selection['code_hash'] != source_code_hash():
        raise ValueError('Selection is stale: manifest or implementation changed')
    if selection['pilot'] and not pilot:
        raise ValueError('Pilot validation needs --pilot; cannot label it final')
    if set(selection['methods']) != set(METHODS) and not pilot:
        raise ValueError('Final protocol requires identity and all seven enhancers')
    checkpoints = []
    seeds = []
    for path, expected in selection['checkpoints'].items():
        if digest(path) != expected:
            raise ValueError('Checkpoint changed since validation')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        seeds.append(saved['seed'])
        checkpoints.append({'path': path, 'sha256': expected, 'seed': saved['seed']})
    if not pilot and (set(seeds) != {11, 22, 33} or set(config['noise_seeds']) != {101, 202, 303}):
        raise ValueError('Final protocol requires three training and three noise seeds')
    for method, params in selection['enhancers'].items():
        if provenance(method, params) != selection['enhancer_provenance'][method]:
            raise ValueError(f'Enhancer changed since validation: {method}')
        config['enhancers'][method] = params
    if not pilot and set(config['conditions']) != {'mild', 'medium', 'severe'}:
        raise ValueError('Final protocol requires all three degradation conditions')
    test = subset_rows(rows, 'test', per_class, subset_seed)
    runs = []
    for checkpoint in checkpoints:
        runs.append({'condition': 'clean', 'noise_seed': '', 'method': 'clean', 'train_seed': checkpoint['seed']})
        for noise in config['noise_seeds']:
            for condition in config['conditions']:
                for method in selection['methods']:
                    runs.append({'condition': condition, 'noise_seed': noise, 'method': method, 'train_seed': checkpoint['seed']})
    snapshot = {'protocol': 'A', 'pilot': pilot, 'config': config, 'manifest_sha256': manifest_hash,
                'selection_sha256': digest(selection_path), 'selection': selection,
                'sample_ids': [r['sample_id'] for r in test], 'subset_seed': subset_seed,
                'subset_id': 'official_test' if per_class is None else config_id([r['sample_id'] for r in test]),
                'checkpoints': checkpoints, 'runs': runs, 'code_hash': source_code_hash(),
                'enhancer_provenance': {m: provenance(m, config['enhancers'][m]) for m in selection['methods']},
                'environment': environment()}
    write_json(output, snapshot)
    return snapshot


def evaluate(root, protocol_path, output, device):
    root, output = Path(root), Path(output)
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    config = protocol['config']
    rows, manifest_hash = manifest_rows(root)
    if manifest_hash != protocol['manifest_sha256'] or source_code_hash() != protocol['code_hash']:
        raise ValueError('Frozen manifest or implementation changed')
    by_id = {r['sample_id']: r for r in rows if r['split'] == 'test'}
    if len(set(protocol['sample_ids'])) != len(protocol['sample_ids']) or not set(protocol['sample_ids']) <= by_id.keys():
        raise ValueError('Invalid frozen test subset')
    rows = [by_id[k] for k in protocol['sample_ids']]
    models = {}
    for c in protocol['checkpoints']:
        if digest(c['path']) != c['sha256']:
            raise ValueError('Frozen checkpoint changed')
        models[c['seed']] = load_cnn(c['path'], manifest_hash, device)[0]
    for method, recorded in protocol['enhancer_provenance'].items():
        if provenance(method, config['enhancers'][method]) != recorded:
            raise ValueError(f'Frozen enhancer changed: {method}')
    lock = output / 'protocol.json'
    if lock.exists() and digest(lock) != digest(protocol_path):
        raise ValueError('Output directory belongs to a different protocol')
    if not lock.exists():
        write_json(lock, protocol)
    states = [dict(run, status='planned') for run in protocol['runs']]
    summaries = []
    for index, run in enumerate(protocol['runs']):
        run_id = config_id(run)
        states[index]['status'] = 'running'
        write_json(output / 'runs.json', states)
        try:
            params = config['enhancers'].get(run['method'], {})
            cached = cache_images(root, rows, manifest_hash, config, run['condition'], run['noise_seed'], run['method'], params, device)
            predictions = predict(root, rows, cached, models[run['train_seed']], config, device)
            write_csv(output / 'predictions' / f'{run_id}.csv', predictions)
            score = classification([r['class_id'] for r in predictions], [r['prediction'] for r in predictions])
            write_json(output / 'classes' / f'{run_id}.json', score)
            summaries.append({'dataset': 'gtsrb', 'split': 'test', 'protocol': 'A', 'subset_id': protocol['subset_id'],
                              **run, 'run_id': run_id, 'config_id': config_id(params), 'n_images': len(rows),
                              'accuracy': score['accuracy'], 'macro_f1': score['macro_f1'],
                              'psnr_mean': float(np.mean([r['psnr'] for r in predictions])),
                              'ssim_mean': float(np.mean([r['ssim'] for r in predictions])),
                              'latency_ms': float(np.mean([r['latency_ms'] for r in predictions]))})
            states[index]['status'] = 'completed'
            write_csv(output / 'summary.csv', summaries)
            print(f"{index+1}/{len(states)} {run}: macro_f1={score['macro_f1']:.4f}", flush=True)
        except Exception as error:
            states[index].update(status='failed', error=str(error))
            write_json(output / 'runs.json', states)
            raise
    write_json(output / 'runs.json', states)
    write_json(output / 'environment.json', environment())
    return summaries
