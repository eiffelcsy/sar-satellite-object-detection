"""Multi-layer ViT fusion and the detection-head selection (CPU only)."""
import pytest
import torch
from torch import nn

from sarbench.model import LayerFusion, MultiTaskModel


class _Block(nn.Module):
    def forward(self, x):
        return x


class _StubBackbone(nn.Module):
    """Mimics the ViT interface: blocks, num_prefix_tokens, token output for a 128 px input (8x8 = 64 tokens)."""

    def __init__(self, dim=128, layers=4):
        super().__init__()
        self.embed_dim = dim
        self.in_chans = 3
        self.num_prefix_tokens = 1
        self._blocks = nn.ModuleList([_Block() for _ in range(layers)])

    @property
    def blocks(self):
        return self._blocks

    def forward(self, x):
        tokens = torch.randn(x.shape[0], 1 + 64, self.embed_dim)
        for block in self._blocks:
            tokens = block(tokens)
        return tokens[:, 1:]


IMAGES = torch.randn(2, 3, 128, 128)
TARGETS = [{'boxes': torch.tensor([[10., 12., 28., 40.]]), 'labels': torch.tensor([1])},
           {'boxes': torch.tensor([[30., 30., 50., 60.]]), 'labels': torch.tensor([2])}]


def test_layer_fusion_preserves_shape_and_is_a_softmax_weighted_sum():
    fusion = LayerFusion(4)
    assert fusion.weights.shape == (4,)
    tokens = [torch.randn(2, 5, 8) for _ in range(4)]
    assert fusion(tokens).shape == (2, 5, 8)
    assert torch.isclose(fusion.weights.softmax(0).sum(), torch.tensor(1.0))


@pytest.mark.parametrize('head', ['standard', 'deform', 'cascade'])
def test_head_variants_forward_backward_and_fuse(head):
    model = MultiTaskModel(_StubBackbone(), task_routing=False, num_classes=9, detail_stem=True,
                           fusion_layers=[1, 2, 3], head=head)
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert model.fusion.weights.grad is not None  # the fusion weights are trained

    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(IMAGES)
    assert logits.shape == (2, 9) and losses == {} and len(detections) == 2
