"""The P2 conv stem, input adapter, WST stem, multi-layer fusion and both RoI heads, end to end (CPU only)."""
import pytest
import torch
from torch import nn

from sarbench.model import ConvStem, InputAdapter, LayerFusion, MultiTaskModel, WSTStem


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


class _Block(nn.Module):
    def forward(self, x):
        return x


class _StubBackbone(nn.Module):
    """The DINOv3 interface the model uses: embed_dim, in_chans, prefix tokens, blocks and a token forward."""

    def __init__(self, layers=4):
        super().__init__()
        self.embed_dim = 128
        self.in_chans = 3
        self.num_prefix_tokens = 1
        self.patch = nn.Conv2d(3, 128, 16, stride=16)  # a stand-in for a patch embedding
        self._blocks = nn.ModuleList([_Block() for _ in range(layers)])

    @property
    def blocks(self):
        return self._blocks

    def unfreeze_patch_embed(self):
        self.patch.requires_grad_(True)

    def forward(self, x):
        tokens = torch.cat([torch.zeros(x.shape[0], 1, self.embed_dim), self.patch(x).flatten(2).transpose(1, 2)], 1)
        for block in self._blocks:
            tokens = block(tokens)
        return tokens[:, 1:]


IMAGES = torch.randn(2, 3, 128, 128)
TARGETS = [{'boxes': torch.tensor([[10., 12., 28., 40.]]), 'labels': torch.tensor([1])},
           {'boxes': torch.tensor([[30., 30., 50., 60.]]), 'labels': torch.tensor([2])}]
CASCADE = {'num_stages': 3, 'share_head': False, 'hidden_dim': 64, 'fg_iou_thresholds': [0.4, 0.5, 0.6],
           'bg_iou_threshold': 0.4, 'stage_loss_weights': [0.5, 0.25, 0.25]}


def test_layer_fusion_preserves_shape_and_is_a_softmax_weighted_sum():
    fusion = LayerFusion(4)
    tokens = [torch.randn(2, 5, 8) for _ in range(4)]
    assert fusion(tokens).shape == (2, 5, 8)
    assert torch.isclose(fusion.weights.softmax(0).sum(), torch.tensor(1.0))


@pytest.mark.parametrize('head, params', [('standard', {}), ('cascade', CASCADE)])
def test_heads_train_with_giou_and_fusion_then_predict(head, params):
    model = MultiTaskModel(_StubBackbone(), num_classes=9, detail_stem=True, fusion_layers=[1, 2, 3],
                           giou_weight=1.0, head=head, head_params=params)
    model.train()
    _, _, losses = model(IMAGES, TARGETS)
    assert set(losses) == {'loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg'}
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert model.fusion.weights.grad is not None  # the fusion weights are trained
    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(IMAGES)
    assert logits.shape == (2, 9) and losses == {} and len(detections) == 2


def test_unknown_head_is_rejected():
    with pytest.raises(ValueError):
        MultiTaskModel(_StubBackbone(), head='deform')


def test_model_with_detail_stem_runs():
    model = MultiTaskModel(_StubBackbone(), num_classes=9, detail_stem=True)
    assert isinstance(model.detail_stem, ConvStem)
    assert sum(p.numel() for p in model.detail_stem.parameters()) > 0
    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 9) and losses == {} and len(detections) == 2


def test_model_with_input_adapter_and_wst_stem_runs():
    model = MultiTaskModel(_StubBackbone(), num_classes=9,
                           input_adapter=True, wst_stem=True)
    assert isinstance(model.input_adapter, InputAdapter) and isinstance(model.wst_stem, WSTStem)
    assert model.backbone.patch.weight.requires_grad  # the adapter unfreezes the patch embedding
    model.eval()
    with torch.no_grad():
        logits, detections, losses = model(torch.randn(2, 3, 512, 512))
    assert logits.shape == (2, 9) and len(detections) == 2
