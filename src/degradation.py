import numpy as np
from .common import stable_seed, validate_image


def degrade(image, gamma_dark=1.0, poisson_peak=None, gaussian_sigma=0.0,
            noise_seed=101, sample_id='', condition=''):
    image = validate_image(image)
    values = [gamma_dark, gaussian_sigma] + ([] if poisson_peak is None else [poisson_peak])
    if not np.isfinite(values).all() or gamma_dark <= 0 or gaussian_sigma < 0 or (poisson_peak is not None and poisson_peak <= 0):
        raise ValueError('Invalid degradation parameters')
    rng = np.random.default_rng(stable_seed(noise_seed, sample_id, condition))
    result = image ** gamma_dark
    if poisson_peak is not None:
        result = rng.poisson(poisson_peak * result) / poisson_peak
    if gaussian_sigma:
        result = result + rng.normal(0, gaussian_sigma, image.shape)
    return np.clip(result, 0, 1).astype(np.float32)
