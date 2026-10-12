"""sarbench/preprocess.py: percentile clipping, the speckle estimate and SAR-BM3D despeckling."""
import numpy as np
import pytest

from sarbench.preprocess import estimate_noise_std, sar_bm3d_despeckle, sigma_clip


def test_sigma_clip_clips_the_top_half_percent_and_scales_to_unit_range():
    x = np.linspace(0, 1, 4000, dtype=np.float32).reshape(100, 40)
    x[0, 0] = 1e6  # a single hyper-bright target pixel
    out = sigma_clip(x, percentile=0.5)
    hi = np.percentile(x, 99.5)
    assert out.min() == 0.0 and out.max() == 1.0
    below = x < hi
    np.testing.assert_allclose(out[below], (x[below] - x.min()) / (hi - x.min()), atol=1e-5)


def test_estimate_noise_std_recovers_the_log_speckle_deviation():
    rng = np.random.default_rng(0)
    speed = rng.gamma(8, 1 / 8, size=(128, 128))
    truth = np.std(np.log(speed))
    estimate = estimate_noise_std(np.log(speed.astype(np.float32)))
    assert abs(estimate - truth) / truth < 0.1


def test_despeckle_reduces_the_variance_of_a_uniform_field():
    pytest.importorskip('bm3d')
    rng = np.random.default_rng(0)
    speed = rng.gamma(8, 1 / 8, size=(64, 64)).astype(np.float32)
    noisy = np.clip(0.5 * speed, 0, 1)
    out = sar_bm3d_despeckle(noisy)
    assert out.var() < noisy.var() / 5  # speckle is mostly gone
    assert abs(float(out.mean()) - 0.5) < 0.1  # the mean reflectivity is preserved
    assert out.min() >= 0 and out.max() <= 1


def test_despeckle_is_deterministic_and_bounded():
    pytest.importorskip('bm3d')
    rng = np.random.default_rng(1)
    img = np.clip(rng.gamma(4, 1 / 4, size=(64, 64)).astype(np.float32), 0, 1)
    first, second = sar_bm3d_despeckle(img), sar_bm3d_despeckle(img)
    np.testing.assert_array_equal(first, second)
    assert first.shape == img.shape and first.dtype == np.float32
    assert first.min() >= 0.0 and first.max() <= 1.0
