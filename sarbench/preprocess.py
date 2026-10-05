"""SAR image pre-processing: SAR-BM3D despeckling and intensity normalisation.

It follows the SAR-BM3D idea (Parrilli et al., "A Nonlocal SAR Image Denoising Algorithm
Based on LLMMSE Wavelet Shrinkage", IEEE TGRS 2012): multiplicative speckle is turned
into additive noise with a homomorphic (log) transform, the compiled bm3d package
(Mäkinen & Foi, Tampere; non-commercial licence) removes it, and the inverse transform
restores the intensity scale.

BM3D expects additive white Gaussian noise and, through its profile, input scaled to
[0, 1]; the log image is therefore rescaled to [0, 1] and the noise std divided by the
same range. The LLMMSE shrinkage of SAR-BM3D reduces to BM3D's DCT hard-thresholding /
Wiener filtering, which is exact for the constant-variance log-speckle model.

Everything operates on float32 arrays with shape (H, W) and values in [0, 1].
"""
import numpy as np

PROFILE, THREADS = 'np', 1  # default bm3d profile and threads per worker


def estimate_noise_std(x):
    """Robust speckle standard deviation of a log-domain image, from the median absolute
    deviation of horizontal/vertical neighbour differences (edges do not bias a median)."""
    dx = np.diff(x, axis=1).ravel()
    dy = np.diff(x, axis=0).ravel()
    d = np.concatenate([dx, dy]).astype(np.float64)
    mad = np.median(np.abs(d - np.median(d)))
    return float(max(1.4826 * mad / np.sqrt(2.0), 1e-4))


def _denoise_library(log_img, sigma, profile=PROFILE, threads=THREADS):
    """Run BM3D from the compiled bm3d package on the log image. The profile assumes
    [0, 1] input, so the image is rescaled to [0, 1] and the noise std divided by the
    same range."""
    try:
        import bm3d as bm3d_lib
    except ImportError as error:
        raise ImportError("SAR-BM3D despeckling needs the 'bm3d' package (pip install bm3d)") from error
    lo, hi = float(log_img.min()), float(log_img.max())
    span = max(hi - lo, 1e-6)
    z = ((log_img - lo) / span).astype(np.float32)
    selector = getattr(bm3d_lib, '_select_profile', None)
    pro = selector(profile) if selector else profile
    if threads is not None and hasattr(pro, 'num_threads'):
        pro.num_threads = int(threads)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):  # benign bm4d profile warnings
        filtered = bm3d_lib.bm3d(z, float(sigma) / span, pro)
    return filtered * span + lo


def sar_bm3d_despeckle(img, sigma=None, profile=PROFILE, threads=THREADS):
    """Despeckle a [0, 1] intensity image with SAR-BM3D. `sigma` is the log-speckle standard
    deviation; when None it is estimated from the image."""
    img = np.clip(np.asarray(img, dtype=np.float32), 1e-4, 1.0)
    log_img = np.log(img)
    if sigma is None:
        sigma = estimate_noise_std(log_img)
    filtered = _denoise_library(log_img, float(sigma), profile, threads)
    return np.clip(np.exp(filtered), 0.0, 1.0).astype(np.float32)


def log_transform(img, floor=1.0 / 255.0):
    """Logarithmic (dB) transform: I_db = 10 * log10(I). Returns dB values <= 0."""
    return (10.0 * np.log10(np.maximum(np.asarray(img, dtype=np.float32), floor))).astype(np.float32)


def sigma_clip(img, percentile=0.5):
    """Clip the brightest `percentile`% of an image to the (100 - percentile)th percentile,
    then scale linearly to [0, 1]. Robust to a few hyper-bright target pixels."""
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(img.min()), float(np.percentile(img, 100.0 - percentile))
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


class SARPreprocess:
    """Compose despeckling, the dB transform and percentile clipping into one callable.

    The order is fixed and matters: despeckle the intensity image, compress its dynamic
    range with the log transform, then clip and rescale to [0, 1] for the network.
    """

    def __init__(self, despeckle=True, log=True, clip_percentile=0.5, sigma=None,
                 profile=PROFILE, threads=THREADS):
        self.despeckle = despeckle
        self.log = log
        self.clip_percentile = clip_percentile
        self.sigma = sigma
        self.profile = profile
        self.threads = threads

    def cache_key(self):
        """Stable name for the configuration, used to key the optional on-disk cache."""
        return (f'd{int(self.despeckle)}_l{int(self.log)}_c{self.clip_percentile}'
                f'_s{self.sigma}_p{self.profile}_t{self.threads}')

    def __call__(self, img):
        img = np.asarray(img, dtype=np.float32)
        if self.despeckle:
            img = sar_bm3d_despeckle(img, sigma=self.sigma, profile=self.profile, threads=self.threads)
        if self.log:
            img = log_transform(img)
        if self.clip_percentile is not None:
            img = sigma_clip(img, self.clip_percentile)
        return img
