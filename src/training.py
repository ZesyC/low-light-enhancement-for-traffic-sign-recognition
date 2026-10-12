import json
from pathlib import Path
import random
import time
import sys
try:
    import resource
except ImportError:  # Windows has no stdlib process RSS counter.
    resource = None

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from .common import digest, environment, read_csv, read_image, stable_seed, synchronize, write_csv, write_json
from .data import subset_rows
from .degradation import degrade
from .metrics import classification
from .model import make_cnn, image_tensor, cnn_config


class CropDataset(Dataset):
    def __init__(self, root, rows, transform=None):
        self.root, self.rows = Path(root), rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tensor = image_tensor(read_image(self.root / row['crop_path']))
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, int(row['class_id'])


class LowLightDataset(CropDataset):
    """Train with epoch-specific noise; validate with separate, fixed noise."""
    def __init__(self, root, rows, config, seed, condition, transform=None):
        super().__init__(root, rows, transform)
        self.config, self.seed, self.condition, self.epoch = config, seed, condition, 0

    def __getitem__(self, index):
        tensor, label = super().__getitem__(index)
        condition = self.condition
        noise = self.config['training']['validation_noise_seed']
        sample_id = self.rows[index]['sample_id']
        if condition == 'mixed':
            noise = stable_seed(self.config['training']['train_noise_seed'], self.seed, self.epoch)
            rng = np.random.default_rng(stable_seed(noise, sample_id, 'condition'))
            condition = ('clean' if rng.random() < self.config['training']['clean_probability']
                         else rng.choice(list(self.config['conditions'])))
        if condition != 'clean':
            image = tensor.permute(1, 2, 0).numpy()
            tensor = image_tensor(degrade(image, **self.config['conditions'][condition],
                                         sample_id=sample_id, noise_seed=noise, condition=condition))
        return tensor, label


TRAIN_TRANSFORM = T.Compose([
    T.RandomRotation(degrees=10),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.RandomErasing(p=0.1, scale=(0.02, 0.08)),
])


def train(root, config, seed, device, output, per_class=None, epochs=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f'cnn_{seed}.pt'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    cfg = cnn_config(config['cnn'])
    training = config.get('training', {'regime': 'clean'})
    if training['regime'] not in {'clean', 'mixed_lowlight'}:
        raise ValueError('Unknown training regime')
    mixed = training['regime'] == 'mixed_lowlight'
    if mixed and (not 0 <= training['clean_probability'] <= 1
                  or not 0 <= training['clean_min_macro_f1'] <= 1
                  or training['train_noise_seed'] == training['validation_noise_seed']
                  or {training['train_noise_seed'], training['validation_noise_seed']} & set(config['noise_seeds'])):
        raise ValueError('Invalid mixed training probabilities, clean guardrail or noise seeds')
    count_epochs = cfg['epochs'] if epochs is None else epochs
    if count_epochs < 1 or cfg['batch_size'] < 1 or cfg['patience'] < 1 or cfg['learning_rate'] <= 0:
        raise ValueError('Invalid training configuration')
    manifest = root / 'data/manifests/gtsrb_samples.csv'
    meta = json.loads(manifest.with_suffix('.json').read_text())
    if digest(manifest) != meta['sha256']:
        raise ValueError('Manifest has changed')
    audit_path = manifest.parent / 'audit.json'
    if not audit_path.exists() or json.loads(audit_path.read_text())['status'] != 'completed':
        raise ValueError('Data audit is missing or failed; run check-data')
    rows = read_csv(manifest)
    train_rows = subset_rows(rows, 'train', per_class)
    validation_rows = subset_rows(rows, 'validation', per_class)
    metadata = {'seed': seed, 'config': config, 'manifest_sha256': digest(manifest),
                'environment': environment(), 'device': str(device),
                'requested_epochs': count_epochs,
                'pilot': per_class is not None or count_epochs != cfg['epochs'],
                'train_ids': [r['sample_id'] for r in train_rows],
                'validation_ids': [r['sample_id'] for r in validation_rows]}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        log = output / f'train_{seed}.json'
        recorded = json.loads(log.read_text()) if log.exists() else {}
        matches = (cnn_config(saved.get('cnn_config')) == cfg
                   and saved.get('manifest_sha256') == metadata['manifest_sha256']
                   and saved.get('seed') == seed and recorded.get('status') == 'completed'
                   and recorded.get('checkpoint_sha256') == digest(checkpoint)
                   and all(recorded.get(k) == metadata[k] for k in
                           ['config', 'requested_epochs', 'pilot', 'train_ids', 'validation_ids']))
        if not matches:
            raise ValueError('Existing checkpoint metadata/config/manifest mismatch or incomplete training; use a new --output')
        print(f'{checkpoint} verified; skipping seed {seed}', flush=True)
        return checkpoint
    loaders = {s: DataLoader(CropDataset(root, r, transform=TRAIN_TRANSFORM if s == 'train' else None),
                            batch_size=cfg['batch_size'], shuffle=s == 'train',
                            num_workers=0, generator=torch.Generator().manual_seed(seed))
               for s, r in [('train', train_rows), ('validation', validation_rows)]}
    if mixed:
        datasets = {'train': LowLightDataset(root, train_rows, config, seed, 'mixed', TRAIN_TRANSFORM),
                    'validation': LowLightDataset(root, validation_rows, config, seed, 'clean')}
        datasets.update({f'validation_{c}': LowLightDataset(root, validation_rows, config, seed, c)
                         for c in config['conditions']})
        loaders = {s: DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=s == 'train',
                                num_workers=0, generator=torch.Generator().manual_seed(seed))
                   for s, dataset in datasets.items()}
    model = make_cnn(cfg, pretrained=True).to(device)
    optimizer_class = torch.optim.AdamW if cfg['optimizer'] == 'adamw' else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history, best, stopping_best, stale = [], -1, -1, 0
    write_json(output / f'train_{seed}.json', dict(metadata, status='running'))
    for epoch in range(1, count_epochs+1):
        if mixed:
            loaders['train'].dataset.epoch = epoch
        synchronize(device)
        epoch_start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        validation_scores = {}
        for split, loader in loaders.items():
            model.train(split == 'train')
            labels, predictions, total_loss = [], [], 0.0
            with torch.set_grad_enabled(split == 'train'):
                for images, targets in loader:
                    images, targets = images.to(device), targets.to(device)
                    logits = model(images)
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite CNN loss')
                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    labels.extend(targets.cpu().tolist())
                    predictions.extend(logits.detach().argmax(1).cpu().tolist())
                    total_loss += float(loss.detach()) * len(targets)
            score = classification(labels, predictions)
            history.append({'epoch': epoch, 'split': split, 'loss': total_loss/len(labels),
                            'accuracy': score['accuracy'], 'macro_f1': score['macro_f1'],
                            'weighted_f1': score['weighted_f1']})
            if split.startswith('validation'):
                validation_scores[split] = score
                print(f"seed={seed} epoch={epoch} val_loss={total_loss/len(labels):.4f} macro_f1={score['macro_f1']:.4f} split={split}", flush=True)
        clean_score = validation_scores['validation']
        selection_f1 = float(np.mean([s['macro_f1'] for s in validation_scores.values()]))
        eligible = not mixed or clean_score['macro_f1'] >= training['clean_min_macro_f1']
        if mixed:
            print(f'seed={seed} epoch={epoch} mean_validation_macro_f1={selection_f1:.4f} clean_guardrail={eligible}', flush=True)
        if eligible and selection_f1 > stopping_best + cfg['min_delta']:
            stopping_best, stale = selection_f1, 0
        else:
            stale += 1
        if eligible and selection_f1 > best:
            best = selection_f1
            temporary = checkpoint.with_suffix('.tmp')
            torch.save({'state_dict': model.state_dict(), 'seed': seed, 'epoch': epoch,
                        'macro_f1': best, 'manifest_sha256': digest(manifest),
                        'cnn_config': cfg, 'pilot': metadata['pilot'], 'training': training,
                        'environment': metadata['environment']}, temporary)
            temporary.replace(checkpoint)
            saved_score = dict(clean_score)
            if mixed:
                saved_score.update(selection_macro_f1=selection_f1, by_condition=validation_scores,
                                   clean_guardrail_passed=True)
            write_json(output / f'validation_{seed}.json', saved_score)
        synchronize(device)
        for entry in history[-len(loaders):]:
            entry['epoch_seconds'] = time.perf_counter() - epoch_start
            entry['peak_rss_bytes'] = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                       * (1 if sys.platform == 'darwin' else 1024)) if resource else None
            entry['accelerator_bytes'] = (torch.cuda.max_memory_allocated(device) if device.type == 'cuda'
                                          else torch.mps.driver_allocated_memory() if device.type == 'mps' else 0)
        write_csv(output / f'learning_{seed}.csv', history)
        if best >= 0 and stale >= cfg['patience']:
            print(f"seed={seed} early stop at epoch={epoch}: patience={stale}, min_delta={cfg['min_delta']}", flush=True)
            break
    if not checkpoint.exists():
        write_json(output / f'train_{seed}.json', dict(metadata, status='failed',
                   reason='No checkpoint meets the clean validation guardrail'))
        raise ValueError('No checkpoint meets the clean validation guardrail; no test evaluation allowed')
    write_json(output / f'train_{seed}.json', dict(metadata, status='completed',
               checkpoint_sha256=digest(checkpoint), best_macro_f1=best))
    return checkpoint
