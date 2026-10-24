"""The DINOv3 backbone on the real checkpoint: [2, 3, 512, 512] images -> the final norm's [2, 1024, 768] patch tokens,
the projections DoRA wraps, and an exact no-op at init for every capacity option. The weights are gated on Hugging
Face, so these run only with SARBENCH_TEST_DINOV3=1 (and a `huggingface-cli login` or a local cache)."""
import os

import pytest
import torch

from sarbench.adapters import adapt_backbone
from sarbench.backbones import DINOv3

pytestmark = pytest.mark.skipif(not os.environ.get('SARBENCH_TEST_DINOV3'),
                                reason='set SARBENCH_TEST_DINOV3=1; the DINOv3 weights are gated on Hugging Face')


def test_dinov3_patch_tokens_with_pretrained_weights():
    backbone = DINOv3(pretrained=True).eval()
    normed = []
    backbone.get_submodule('dinov3.norm').register_forward_hook(lambda module, args, out: normed.append(out))
    with torch.no_grad():
        tokens = backbone(torch.randn(2, 3, 512, 512))
    assert tokens.shape == (2, 1024, 768)  # 32 x 32 patches, prefix tokens dropped
    assert tokens.isfinite().all()
    assert torch.equal(tokens, normed[0][:, -1024:])
    assert len(backbone.blocks) == 12 and backbone.embed_dim == 768 and backbone.num_prefix_tokens == 5


def test_adapter_targets_cover_attention_and_mlp_of_every_block():
    targets = DINOv3(pretrained=False).adapter_targets(('attn', 'mlp'))
    assert len(targets) == 12 * 6 and {index for index, _, _ in targets} == set(range(12))


@pytest.mark.parametrize('options', [
    {},
    {'full_blocks': [0, 1]},
    {'rank_overrides': {0: 64}},
    {'conv_layers': [0, 5], 'conv_dim': 32},
])
def test_capacity_options_are_noops_at_init(options):
    backbone = DINOv3(pretrained=True).eval()
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        before = backbone(x)
        adapt_backbone(backbone, **options)
        after = backbone(x)
    assert (after - before).abs().max() < 1e-3  # DoRA's weight renormalization adds fp32 noise


def test_patch8_tokens_equal_16px_tokens_of_each_patch_upsampled():
    """PI-resize: an 8 px token is the pretrained token of that 8 x 8 patch bilinearly resized to 16 x 16."""
    import torch.nn.functional as F
    fine, coarse = DINOv3(pretrained=True, patch_size=8).eval(), DINOv3(pretrained=True).eval()
    assert fine.patch_size == 8 and fine.dinov3.embeddings.patch_embeddings.weight.shape[-1] == 8
    x = torch.rand(1, 3, 64, 64)
    patches = x.unfold(2, 8, 8).unfold(3, 8, 8)  # [1, 3, 8, 8, 8, 8]: grid then pixels
    upsampled = F.interpolate(patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 8, 8), size=16, mode='bilinear',
                              align_corners=False)
    upsampled = upsampled.view(1, 8, 8, 3, 16, 16).permute(0, 3, 1, 4, 2, 5).reshape(1, 3, 128, 128)
    with torch.no_grad():
        tokens = fine(x)
        assert tokens.shape == (1, 64, 768)
        torch.testing.assert_close(tokens, coarse(upsampled), atol=1e-3, rtol=1e-3)
