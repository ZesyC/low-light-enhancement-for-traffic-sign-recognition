"""RGB-only enhancement. No adapter accepts a reference image or label."""
import argparse
import importlib.util
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .common import digest, git_commit, stable_seed, validate_image
from .model import image_tensor

DEEP = {'zero_dce', 'retinexformer', 'diffusion'}
METHODS = ('identity', 'gamma', 'he', 'clahe', 'retinex', 'zero_dce', 'retinexformer', 'diffusion')


def classical(image, method, params=None):
    image = validate_image(image)
    params = params or {}
    if method == 'identity':
        return image.copy()
    if method == 'gamma':
        gamma = params.get('gamma', 0.6)
        if not 0 < gamma < 1:
            raise ValueError('Enhancement gamma must be in (0,1)')
        return image ** gamma
    if method in {'he', 'clahe'}:
        ycc = cv2.cvtColor(np.rint(image * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        if method == 'he':
            ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
        else:
            clip, tiles = params.get('clip_limit', 2.0), params.get('tile_grid', [8, 8])
            if not np.isfinite(clip) or clip <= 0 or len(tiles) != 2 or any(not isinstance(t, int) or t < 1 for t in tiles):
                raise ValueError('Invalid CLAHE parameters')
            ycc[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=tuple(tiles)).apply(ycc[..., 0])
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255
    if method == 'retinex':
        sigmas = params.get('sigmas', [3, 10, 20])
        if not sigmas or not np.isfinite(sigmas).all() or min(sigmas) <= 0:
            raise ValueError('MSR sigmas must be finite and positive')
        result = sum(np.log(image + 1e-6) - np.log(cv2.GaussianBlur(image, (0, 0), sigmaX=s) + 1e-6)
                     for s in sigmas) / len(sigmas)
        lo, hi = np.percentile(result, [1, 99])
        return image.copy() if hi - lo < 1e-6 else np.clip((result-lo)/(hi-lo), 0, 1).astype(np.float32)
    raise ValueError(f'Unknown classical enhancer: {method}')


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace(value):
    return argparse.Namespace(**{k: namespace(v) if isinstance(v, dict) else v for k, v in value.items()})


def provenance(method, params):
    metadata = {'method': method, 'params': params, 'precision': 'float32', 'input': 'RGB [0,1]',
                'reference_access': False, 'output_storage': 'float32 npy; no further quantization'}
    if method in DEEP:
        checkpoint, repo = Path(params['checkpoint']), Path(params['repo'])
        if not checkpoint.is_file():
            raise FileNotFoundError(f'{method}: missing author checkpoint {checkpoint}; see README.md')
        commit = git_commit(repo)
        if commit is None:
            raise ValueError(f'{method}: author repository must have a recorded git commit')
        metadata.update(checkpoint_sha256=digest(checkpoint), repo_commit=commit)
        code_files = {'zero_dce': ['Zero-DCE_code/model.py'],
                      'retinexformer': ['basicsr/models/archs/RetinexFormer_arch.py'],
                      'diffusion': ['models/ddm.py', 'models/unet.py', 'models/mods.py', 'models/wavelet.py']}[method]
        metadata['code_sha256'] = {f: digest(repo / f) for f in code_files}
        if 'config' in params:
            metadata['config_sha256'] = digest(params['config'])
    return metadata


class Enhancer:
    def __init__(self, method, params, device):
        if method not in METHODS:
            raise ValueError(f'Unknown method: {method}')
        self.method, self.params, self.device = method, params, device
        self.metadata = provenance(method, params)
        self.model = None
        if method not in DEEP:
            return
        repo = Path(params['repo']).resolve()
        # Load only author architectures, avoiding their dataset/GT-dependent evaluation scripts.
        if method == 'zero_dce':
            module = import_file('author_zero_dce', repo / 'Zero-DCE_code/model.py')
            self.model = module.enhance_net_nopool()
        elif method == 'retinexformer':
            module = import_file('author_retinexformer', repo / 'basicsr/models/archs/RetinexFormer_arch.py')
            cfg = yaml.safe_load(Path(params['config']).read_text())['network_g'].copy()
            if cfg.pop('type') != 'RetinexFormer':
                raise ValueError('Expected RetinexFormer architecture config')
            self.model = module.RetinexFormer(**cfg)
        else:
            # Author modules use absolute imports (models, utils); do not import other repos that way.
            sys.path.insert(0, str(repo))
            from models.ddm import Net
            cfg = namespace(yaml.safe_load(Path(params['config']).read_text()))
            cfg.device = device
            steps = params['sampling_timesteps']
            if not isinstance(steps, int) or not 1 <= steps <= cfg.diffusion.num_diffusion_timesteps or cfg.diffusion.num_diffusion_timesteps % steps:
                raise ValueError('Sampling steps must divide the author diffusion timestep count')
            self.model = Net(argparse.Namespace(sampling_timesteps=steps), cfg)
        # Legacy diffusion checkpoint contains argparse.Namespace; allow only that safe container.
        with torch.serialization.safe_globals([argparse.Namespace]):
            state = torch.load(params['checkpoint'], map_location='cpu', weights_only=True)
        if method == 'retinexformer':
            state = state['params']
        elif method == 'diffusion':
            state = state['ema_helper'] if params.get('ema', True) else state['state_dict']
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.metadata['padding'] = 'replicate to multiple of 32' if method == 'diffusion' else 'replicate to multiple of 4' if method == 'retinexformer' else 'none'

    @torch.inference_mode()
    def __call__(self, image, sample_id=''):
        image = validate_image(image)
        if self.model is None:
            return validate_image(classical(image, self.method, self.params))
        tensor = image_tensor(image).unsqueeze(0).to(self.device)
        h, w = image.shape[:2]
        multiple = 32 if self.method == 'diffusion' else 4 if self.method == 'retinexformer' else 1
        tensor = F.pad(tensor, (0, (-w) % multiple, 0, (-h) % multiple), mode='replicate')
        if self.method == 'diffusion':
            seed = stable_seed(self.params.get('inference_seed', 230), sample_id) % (2**63-1)
            torch.manual_seed(seed)
        output = self.model(tensor)
        if self.method == 'zero_dce':
            output = output[1]
        elif self.method == 'diffusion':
            output = output['pred_x']
        if output.shape[:2] != (1, 3) or output.shape[2:] != tensor.shape[2:] or not torch.isfinite(output).all():
            raise ValueError(f'{self.method}: malformed/nonfinite output')
        result = output[0, :, :h, :w].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        return validate_image(result)


def validation_grid(method, defaults):
    if method == 'gamma':
        return [{'gamma': g} for g in [0.4, 0.6, 0.8]]
    if method == 'clahe':
        return [{'clip_limit': c, 'tile_grid': t} for c in [1.0, 2.0, 4.0] for t in [[4, 4], [8, 8]]]
    if method == 'retinex':
        return [{'sigmas': s} for s in [[3, 10, 20], [5, 15, 30]]]
    return [defaults]
