"""The real-detail P2 conv stem, the input adapter and the WST stem, and their integration (CPU only)."""
import torch
from torch import nn

from sarbench.model import ConvStem, InputAdapter, MultiTaskModel, WSTStem


def test_conv_stem_has_stride_four_and_the_requested_channels():
    stem = ConvStem(in_chans=3, channels=16, out_channels=64)
    out = stem(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 64, 16, 16)  # two stride-2 convs -> stride 4


def test_wst_stem_is_81_coefficients_at_stride_four():
    stem = WSTStem(out_channels=64, J=2, L=8)
    out = stem(torch.randn(1, 3, 512, 512))
    assert out.shape == (1, 64, 128, 128)  # 512 / 2**2 = 128, i.e. P2's resolution


def test_input_adapter_starts_as_a_no_op():
    adapter = InputAdapter(3)
    x = torch.rand(2, 3, 16, 16)
    normalized = (x - adapter.mean) / adapter.std
    assert torch.allclose(adapter(x), normalized, atol=1e-6)


class _StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 128
        self.in_chans = 3
        self.patch = nn.Conv2d(3, 128, 16, stride=16)  # a stand-in for a patch embedding

    def unfreeze_patch_embed(self):
        self.patch.requires_grad_(True)

    def forward(self, x):
        return torch.randn(x.shape[0], (x.shape[-1] // 16) ** 2, self.embed_dim)


def test_model_with_detail_stem_runs():
    model = MultiTaskModel(_StubBackbone(), task_routing=False, num_classes=9, detail_stem=True)
    assert isinstance(model.detail_stem, ConvStem)
    assert sum(p.numel() for p in model.detail_stem.parameters()) > 0
    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 9) and losses == {} and len(detections) == 2


def test_model_with_input_adapter_and_wst_stem_runs():
    model = MultiTaskModel(_StubBackbone(), task_routing=False, num_classes=9,
                           input_adapter=True, wst_stem=True)
    assert isinstance(model.input_adapter, InputAdapter) and isinstance(model.wst_stem, WSTStem)
    assert model.backbone.patch.weight.requires_grad  # the adapter unfreezes the patch embedding
    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(torch.randn(2, 3, 512, 512))
    assert logits.shape == (2, 9) and len(detections) == 2
