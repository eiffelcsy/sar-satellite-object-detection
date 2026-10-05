"""Pseudo-RGB channel assembly for the DINOv3 backbone: shapes, ranges, edge modes, cache keys (CPU only)."""
import numpy as np

from sarbench import channels
from sarbench.channels import PseudoRGB


def test_pseudo_rgb_stack_shape_and_range():
    img = np.random.RandomState(0).rand(64, 64).astype(np.float32)
    out = PseudoRGB(despeckle=False, edge='sobel')(img)
    assert out.shape == (3, 64, 64) and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_both_edge_modes_highlight_a_square_edge():
    img = np.zeros((32, 32), np.float32)
    img[8:24, 8:24] = 1.0
    for mode in ('sobel', 'highpass'):
        out = PseudoRGB(despeckle=False, edge=mode)(img)
        assert out[2].max() > 0  # the square's edge is visible in channel 3


def test_cache_key_reflects_configuration():
    assert PseudoRGB(despeckle=False, edge='sobel').cache_key() \
        == PseudoRGB(despeckle=False, edge='sobel').cache_key()
    assert PseudoRGB(despeckle=False, edge='sobel').cache_key() \
        != PseudoRGB(despeckle=False, edge='highpass').cache_key()


def test_despeckled_base_uses_sar_bm3d(monkeypatch):
    calls = {}

    def fake_bm3d(img, sigma=None, profile=None, threads=None):
        calls['args'] = (sigma, profile, threads)
        return img

    monkeypatch.setattr(channels, 'sar_bm3d_despeckle', fake_bm3d)
    img = np.random.RandomState(1).rand(16, 16).astype(np.float32)
    out = PseudoRGB(despeckle=True, sigma=0.2, profile='np', threads=3)(img)
    assert out.shape == (3, 16, 16)
    assert calls['args'] == (0.2, 'np', 3)  # channel 2 went through SAR-BM3D with the configured options


def test_as_image_tensor_keeps_channel_count():
    from sarbench.data import _as_image_tensor
    assert _as_image_tensor(np.zeros((4, 4), np.float32)).shape == (1, 4, 4)
    assert _as_image_tensor(np.zeros((3, 4, 4), np.float32)).shape == (3, 4, 4)
