"""Pseudo-RGB channel assembly for single-channel SAR, used by the DINOv3 backbone.

A DINOv3 checkpoint expects three input channels; instead of repeating the gray image, this expands it into
three complementary [0, 1] representations:

    1. normalized amplitude  -- the intensity image with its dynamic range stretched (spatial structure);
    2. despeckled base image -- SAR-BM3D (the same denoiser as sarbench/preprocess.py), or a Gaussian blur
                                when despeckling is off / unavailable (coherent structure without speckle);
    3. edge map              -- a Sobel gradient magnitude (or a Gaussian high-pass) of the base image
                                (high-frequency structure).

The result is a (3, H, W) float32 array, so it plugs into the existing data pipeline as a pre-processing
callable (it exposes `cache_key()`, so `--preprocess-cache` caches the SAR-BM3D pass across epochs).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel

from .preprocess import PROFILE, THREADS, sar_bm3d_despeckle, sigma_clip


def normalized_amplitude(img, percentile=0.5):
    """Channel 1: stretch the intensity image to [0, 1], clipping the brightest `percentile`%."""
    return sigma_clip(img, percentile=percentile)


def despeckled_base(img, despeckle=True, sigma=None, profile=PROFILE, threads=THREADS, blur_sigma=1.0):
    """Channel 2: SAR-BM3D when `despeckle`, otherwise a Gaussian blur as a cheap base image."""
    if despeckle:
        return sar_bm3d_despeckle(img, sigma=sigma, profile=profile, threads=threads)
    return gaussian_filter(np.asarray(img, dtype=np.float32), blur_sigma)


def edge_map(img, mode='sobel', blur_sigma=1.0):
    """Channel 3: Sobel gradient magnitude, or a Gaussian high-pass, of `img`."""
    if mode == 'highpass':
        return np.clip(np.asarray(img, dtype=np.float32) - gaussian_filter(img, blur_sigma), 0.0, None)
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    return np.hypot(gx, gy)


class PseudoRGB:
    """(H, W) intensity image in [0, 1] -> (3, H, W) pseudo-RGB image in [0, 1]."""

    def __init__(self, despeckle=True, edge='sobel', clip_percentile=0.5, sigma=None,
                 profile=PROFILE, threads=THREADS, blur_sigma=1.0):
        self.despeckle = despeckle
        self.edge = edge
        self.clip_percentile = clip_percentile
        self.sigma = sigma
        self.profile = profile
        self.threads = threads
        self.blur_sigma = blur_sigma

    def cache_key(self):
        """Stable name for the configuration, used to key the optional on-disk cache."""
        return (f'prgb_d{int(self.despeckle)}_e{self.edge}_c{self.clip_percentile}_s{self.sigma}'
                f'_p{self.profile}_t{self.threads}_b{self.blur_sigma}')

    def __call__(self, img):
        img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
        amplitude = normalized_amplitude(img, self.clip_percentile)
        base = despeckled_base(img, self.despeckle, self.sigma, self.profile, self.threads, self.blur_sigma)
        base = sigma_clip(base, percentile=self.clip_percentile)
        edges = sigma_clip(edge_map(base, self.edge, self.blur_sigma), percentile=self.clip_percentile)
        return np.stack([amplitude, base, edges]).astype(np.float32)
