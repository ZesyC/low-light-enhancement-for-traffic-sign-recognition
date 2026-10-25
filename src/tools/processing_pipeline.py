"""Denoising and ordered RGB pipelines. No operation receives labels or references."""
import cv2
import numpy as np

from src.common import validate_image, digest
from src.enhancement import Enhancer, provenance
from src.experiment import source_code_hash


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _odd(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value % 2 != 1:
        raise ValueError(f'{name} must be a positive odd integer')


def denoise(image, method, params):
    image = validate_image(image)
    if method == 'gaussian':
        _odd(params['kernel'], 'kernel'); _positive(params['sigma'], 'sigma')
        result = cv2.GaussianBlur(image, (params['kernel'], params['kernel']), params['sigma'], borderType=cv2.BORDER_REFLECT_101)
    elif method == 'bilateral':
        _odd(params['diameter'], 'diameter')
        _positive(params['sigma_color'], 'sigma_color'); _positive(params['sigma_space'], 'sigma_space')
        result = cv2.bilateralFilter(image, params['diameter'], params['sigma_color'], params['sigma_space'], borderType=cv2.BORDER_REFLECT_101)
    elif method == 'nlm':
        _positive(params['h'], 'h'); _positive(params['h_color'], 'h_color')
        _odd(params['template'], 'template'); _odd(params['search'], 'search')
        if params['search'] < params['template']:
            raise ValueError('search must be at least template')
        # OpenCV's color NLM expects uint8 BGR and internally uses Lab.
        bgr = cv2.cvtColor(np.rint(image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        result = cv2.fastNlMeansDenoisingColored(bgr, None, params['h'], params['h_color'], params['template'], params['search'])
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    else:
        raise ValueError(f'Unknown denoiser: {method}')
    if result.shape != image.shape:
        raise ValueError('Denoiser changed image dimensions')
    return validate_image(np.clip(result, 0, 1).astype(np.float32))


class Pipeline:
    def __init__(self, spec, device):
        self.spec = spec
        self.steps = []
        for step in spec['steps']:
            if step['kind'] == 'denoise':
                if step['method'] not in {'gaussian','bilateral','nlm'}:
                    raise ValueError('Unknown denoiser')
                self.steps.append((step, None))
            elif step['kind'] == 'enhance':
                self.steps.append((step, Enhancer(step['method'], step['params'], device)))
            else:
                raise ValueError('Unknown pipeline step kind')

    def __call__(self, image, sample_id=''):
        result = validate_image(image).copy()
        shape = result.shape
        for step, enhancer in self.steps:
            result = denoise(result, step['method'], step['params']) if enhancer is None else enhancer(result, sample_id)
        if result.shape != shape:
            raise ValueError('Pipeline changed dimensions')
        return validate_image(result)


def pipeline_provenance(spec):
    operations = []
    for step in spec['steps']:
        if step['kind'] == 'enhance':
            operations.append(provenance(step['method'], step['params']))
        elif step['kind'] == 'denoise':
            operations.append({'method': step['method'], 'params': step['params'],
                               'quantization': 'uint8 BGR to Lab inside OpenCV' if step['method']=='nlm' else 'float32 RGB',
                               'opencv': cv2.__version__})
        else:
            raise ValueError('Unknown pipeline step')
    return {'spec': spec, 'operations': operations, 'baseline_implementation': source_code_hash(),
            'pipeline_implementation': digest(__file__), 'reference_access': False, 'label_access': False}
