import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def config_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def stable_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256(json.dumps([seed, *parts]).encode()).digest()[:8], 'little')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty result table')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_image(image):
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f'Expected HWC RGB, got {image.shape}')
    if not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all():
        raise ValueError('Expected finite floating RGB')
    if image.min() < 0 or image.max() > 1:
        raise ValueError('RGB must be within [0,1]')
    return image.astype(np.float32, copy=False)


def read_image(path):
    if Path(path).suffix == '.npy':
        return validate_image(np.load(path, allow_pickle=False))
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32) / 255


def save_array(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, validate_image(image), allow_pickle=False)
    temporary.replace(path)


def git_commit(path='.'):
    result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def environment():
    import importlib.metadata
    import torch
    return {'date': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
            'platform': platform.platform(), 'commit': git_commit(),
            'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available(),
            'packages': {p: importlib.metadata.version(p) for p in
                         ['numpy', 'torch', 'pillow', 'opencv-python-headless', 'scikit-image', 'scikit-learn']}}


def device_for(name='auto'):
    import torch
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(name)


def synchronize(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()
