"""The real-detail P2 conv stem and its integration into the multi-task model (CPU only)."""
import torch
from torch import nn

from sarbench.model import ConvStem, MultiTaskModel


def test_conv_stem_has_stride_four_and_the_requested_channels():
    stem = ConvStem(in_chans=3, channels=16, out_channels=64)
    out = stem(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 64, 16, 16)  # two stride-2 convs -> stride 4


class _StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 128
        self.in_chans = 3

    def forward(self, x):
        return torch.randn(x.shape[0], 8 * 8, self.embed_dim)


def test_model_with_detail_stem_runs():
    model = MultiTaskModel(_StubBackbone(), task_routing=False, num_classes=9, detail_stem=True)
    assert isinstance(model.detail_stem, ConvStem)
    assert sum(p.numel() for p in model.detail_stem.parameters()) > 0
    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 9) and losses == {} and len(detections) == 2
