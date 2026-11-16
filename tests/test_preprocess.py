import numpy as np
import pytest

from sarbench.preprocess import PseudoRGB, estimate_noise_std, sigma_clip


def test_sigma_clip_clips_the_brightest_pixels_and_scales_to_unit_range():
    img = np.linspace(0, 1, 1000, dtype=np.float32)
    out = sigma_clip(img, percentile=0.5)
    assert out.min() == 0 and out.max() == 1 and (out == 1).sum() == 5
    assert (sigma_clip(np.full(10, 0.3)) == 0).all()


def test_estimate_noise_std_recovers_the_log_speckle_deviation():
    rng = np.random.default_rng(0)
    assert estimate_noise_std(rng.normal(0, 0.2, (256, 256))) == pytest.approx(0.2, rel=0.05)


def test_pseudo_rgb_is_three_unit_range_channels():
    pytest.importorskip('bm3d')
    rng = np.random.default_rng(0)
    img = np.clip(0.3 * rng.gamma(4, 0.25, (64, 64)), 0, 1).astype(np.float32)
    out = PseudoRGB()(img)
    assert out.shape == (3, 64, 64) and out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1
    assert out[1].std() < out[0].std()  # despeckled: smoother than the amplitude
    assert PseudoRGB().cache_key() == 'prgb_d1_esobel_c0.5_sNone_pnp_t1_b1.0_mdespeckle'  # the training caches' name
