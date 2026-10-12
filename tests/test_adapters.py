"""DoRA, Convpass and the per-block capacity options of `adapt_backbone` (CPU only)."""
import pytest
import torch
from torch import nn

from sarbench.adapters import AdaptedBlock, ConvAdapter, DoRA, adapt_backbone


class _Block(nn.Module):
    """A transformer-block stand-in with the DINOv3 attribute layout (attention q/k/v/o, mlp up/down)."""

    def __init__(self, dim=16):
        super().__init__()
        self.attention = nn.Module()
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            setattr(self.attention, name, nn.Linear(dim, dim))
        self.mlp = nn.Module()
        self.mlp.up_proj, self.mlp.down_proj = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)

    def forward(self, x, scale=1.0):
        a = self.attention
        x = x + a.o_proj(a.q_proj(x) + a.k_proj(x) + a.v_proj(x)) * scale
        return x + self.mlp.down_proj(torch.relu(self.mlp.up_proj(x)))


class _Backbone(nn.Module):
    """The DINOv3 interface adapt_backbone relies on: blocks, embed_dim, num_prefix_tokens, adapter_targets."""

    def __init__(self, blocks=4, dim=16):
        super().__init__()
        self.embed_dim, self.num_prefix_tokens = dim, 1
        self.layer = nn.ModuleList(_Block(dim) for _ in range(blocks))

    @property
    def blocks(self):
        return self.layer

    def adapter_targets(self, groups=('attn', 'mlp')):
        targets = []
        for index, block in enumerate(self.blocks):
            if 'attn' in groups:
                targets += [(index, block.attention, n) for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
            if 'mlp' in groups:
                targets += [(index, block.mlp, n) for n in ('up_proj', 'down_proj')]
        return targets

    def forward(self, x):
        for block in self.blocks:
            x = block(x, scale=0.5)  # a keyword argument, as DINOv3 passes its RoPE embeddings
        return x


def tokens(n=1 + 16, dim=16):
    return torch.randn(2, n, dim, dtype=torch.float64)  # 1 prefix token + a 4 x 4 grid


def test_dora_is_a_noop_at_init_and_trains_its_magnitude():
    base = nn.Linear(16, 8)
    layer = DoRA(base, rank=4, alpha=8).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    # B = 0 and m = ||W||_c -> W' = W, up to the norm's 1e-6 epsilon and m being computed in fp32
    torch.testing.assert_close(layer(x), base(x), atol=1e-5, rtol=1e-5)
    nn.init.normal_(layer.up.weight, std=0.1)
    assert not torch.allclose(layer(x), base(x))
    assert layer.magnitude.requires_grad and layer.down.weight.requires_grad


def test_conv_adapter_is_a_noop_at_init_and_mixes_neighbouring_patches():
    adapter = ConvAdapter(dim=16, hidden=8, num_prefix_tokens=1).double()
    x = tokens()
    torch.testing.assert_close(adapter(x), x)
    nn.init.normal_(adapter.up.weight, std=0.1)
    out = adapter(x)
    torch.testing.assert_close(out[:, 0], x[:, 0])  # the prefix token passes through
    poked = x.clone()
    poked[:, 1 + 5] += 1.0  # one patch of the 4 x 4 grid (row 1, column 1)
    changed = (adapter(poked) - out).abs().sum(-1)[0, 1:].view(4, 4) > 1e-9
    assert changed[0, 0] and changed[2, 2] and not changed[3, 3]  # its 3x3 neighbourhood only (one conv)


def test_adapt_backbone_defaults_wrap_every_projection_and_freeze_the_rest():
    backbone = _Backbone().double()
    x = tokens()
    before = backbone(x)
    adapt_backbone(backbone, rank=4, alpha=4)
    torch.testing.assert_close(backbone(x), before, atol=1e-4, rtol=1e-4)  # a no-op at init (DoRA epsilon)
    assert all(isinstance(getattr(b.attention, n), DoRA) for b in backbone.blocks for n in ('q_proj', 'o_proj'))
    trainable = [n for n, p in backbone.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith(('.down.weight', '.up.weight', '.magnitude')) for n in trainable)


def test_full_blocks_train_entirely_without_dora():
    backbone = _Backbone()
    adapt_backbone(backbone, rank=4, alpha=4, full_blocks=[0, -1])
    for index in (0, 3):
        block = backbone.blocks[index]
        assert isinstance(block.attention.q_proj, nn.Linear) and not isinstance(block.attention.q_proj, DoRA)
        assert all(p.requires_grad for p in block.parameters())
    assert isinstance(backbone.blocks[1].attention.q_proj, DoRA)
    assert not backbone.blocks[1].attention.q_proj.base.weight.requires_grad


def test_rank_overrides_keep_alpha_over_rank():
    backbone = _Backbone()
    adapt_backbone(backbone, rank=4, alpha=8, rank_overrides={1: 12})
    assert backbone.blocks[0].attention.q_proj.down.out_features == 4
    layer = backbone.blocks[1].attention.q_proj
    assert layer.down.out_features == 12 and layer.scale == pytest.approx(2.0)  # alpha / rank unchanged


def test_full_blocks_and_rank_overrides_must_not_overlap():
    with pytest.raises(ValueError):
        adapt_backbone(_Backbone(), full_blocks=[1], rank_overrides={1: 8})


def test_conv_adapters_wrap_blocks_forward_their_arguments_and_start_as_noops():
    backbone = _Backbone().double()
    x = tokens()
    before = backbone(x)
    adapt_backbone(backbone, rank=4, alpha=4, conv_layers=[0, 2], conv_dim=8)
    assert isinstance(backbone.blocks[0], AdaptedBlock) and isinstance(backbone.blocks[2], AdaptedBlock)
    assert isinstance(backbone.blocks[0].block.attention.q_proj, DoRA)  # DoRA was applied before wrapping
    torch.testing.assert_close(backbone.double()(x), before, atol=1e-4, rtol=1e-4)  # 4 DoRA blocks of epsilon
    assert backbone.blocks[0].adapter.conv.weight.requires_grad
