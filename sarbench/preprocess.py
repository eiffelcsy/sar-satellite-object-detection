"""Pseudo-RGB input for the DINOv3 backbone: one gray SAR image -> three complementary [0, 1] channels.

    1. amplitude    the gray image, its brightest `clip_percentile` % clipped, scaled to [0, 1];
    2. despeckled   SAR-BM3D (Parrilli et al., IEEE TGRS 2012): the multiplicative speckle becomes additive in the
                    log domain, where the compiled `bm3d` package (Mäkinen & Foi; non-commercial licence) removes it;
    3. edges        the Sobel gradient magnitude of the despeckled image.

Every function works on float32 (H, W) arrays in [0, 1].
"""
import numpy as np
from scipy.ndimage import sobel


def estimate_noise_std(x):
    """Robust speckle standard deviation of a log-domain image, from the median absolute deviation of horizontal /
    vertical neighbour differences (edges do not bias a median)."""
    d = np.concatenate([np.diff(x, axis=1).ravel(), np.diff(x, axis=0).ravel()]).astype(np.float64)
    mad = np.median(np.abs(d - np.median(d)))
    return float(max(1.4826 * mad / np.sqrt(2.0), 1e-4))


def sar_bm3d_despeckle(img, sigma=None, profile='np', threads=1):
    """Despeckle a [0, 1] intensity image with SAR-BM3D; `sigma` (the log-speckle std) is estimated when None.

    BM3D's profile assumes [0, 1] input, so the log image is rescaled to [0, 1] and the noise std by the same range.
    """
    import bm3d  # lazy: only needed when an image is not in the pre-processing cache
    log_img = np.log(np.clip(np.asarray(img, dtype=np.float32), 1e-4, 1.0))
    if sigma is None:
        sigma = estimate_noise_std(log_img)
    lo, hi = float(log_img.min()), float(log_img.max())
    span = max(hi - lo, 1e-6)
    z = ((log_img - lo) / span).astype(np.float32)
    selector = getattr(bm3d, '_select_profile', None)
    pro = selector(profile) if selector else profile
    if threads is not None and hasattr(pro, 'num_threads'):
        pro.num_threads = int(threads)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):  # benign bm4d profile warnings
        filtered = bm3d.bm3d(z, float(sigma) / span, pro) * span + lo
    return np.clip(np.exp(filtered), 0.0, 1.0).astype(np.float32)


def sigma_clip(img, percentile=0.5):
    """Clip the brightest `percentile` % of an image to the (100 - percentile)th percentile, then scale linearly to
    [0, 1]. Robust to a few hyper-bright target pixels."""
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(img.min()), float(np.percentile(img, 100.0 - percentile))
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


class PseudoRGB:
    """(H, W) intensity image in [0, 1] -> (3, H, W) float32 [amplitude, SAR-BM3D despeckled, Sobel edges]."""

    def __init__(self, clip_percentile=0.5):
        self.clip_percentile = clip_percentile

    def cache_key(self):
        """Names the pre-processing in cache file names (the format of the training runs, so their caches load)."""
        return f'prgb_d1_esobel_c{self.clip_percentile}_sNone_pnp_t1_b1.0_mdespeckle'

    def __call__(self, img):
        img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
        amplitude = sigma_clip(img, self.clip_percentile)
        base = sigma_clip(sar_bm3d_despeckle(img), self.clip_percentile)
        edges = sigma_clip(np.hypot(sobel(base, axis=1), sobel(base, axis=0)), self.clip_percentile)
        return np.stack([amplitude, base, edges]).astype(np.float32)
