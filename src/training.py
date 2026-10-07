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

from .common import digest, environment, read_csv, read_image, synchronize, write_csv, write_json
from .data import subset_rows
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
    model = make_cnn(cfg, pretrained=True).to(device)
    optimizer_class = torch.optim.AdamW if cfg['optimizer'] == 'adamw' else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    history, best, stale = [], -1, 0
    write_json(output / f'train_{seed}.json', dict(metadata, status='running'))
    for epoch in range(1, count_epochs+1):
        synchronize(device)
        epoch_start = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
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
            if split == 'validation':
                print(f"seed={seed} epoch={epoch} val_loss={total_loss/len(labels):.4f} macro_f1={score['macro_f1']:.4f}", flush=True)
                if score['macro_f1'] > best:
                    best, stale = score['macro_f1'], 0
                    temporary = checkpoint.with_suffix('.tmp')
                    torch.save({'state_dict': model.state_dict(), 'seed': seed, 'epoch': epoch,
                                'macro_f1': best, 'manifest_sha256': digest(manifest),
                                'cnn_config': cfg, 'pilot': metadata['pilot'],
                                'environment': metadata['environment']}, temporary)
                    temporary.replace(checkpoint)
                    write_json(output / f'validation_{seed}.json', score)
                else:
                    stale += 1
        synchronize(device)
        for entry in history[-2:]:
            entry['epoch_seconds'] = time.perf_counter() - epoch_start
            entry['peak_rss_bytes'] = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                       * (1 if sys.platform == 'darwin' else 1024)) if resource else None
            entry['accelerator_bytes'] = (torch.cuda.max_memory_allocated(device) if device.type == 'cuda'
                                          else torch.mps.driver_allocated_memory() if device.type == 'mps' else 0)
        write_csv(output / f'learning_{seed}.csv', history)
        if stale >= cfg['patience']:
            break
    write_json(output / f'train_{seed}.json', dict(metadata, status='completed',
               checkpoint_sha256=digest(checkpoint), best_macro_f1=best))
    return checkpoint
