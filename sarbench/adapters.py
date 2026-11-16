"""Backbone adaptation: the DINOv3 ViT is frozen, every attention and MLP projection is wrapped in DoRA, and a
Convpass adapter follows every block. Both start as exact no-ops, so training begins at the pretrained ViT.
"""
import torch
import torch.nn.functional as F
from torch import nn


class DoRA(nn.Module):
    """Weight-decomposed LoRA (Liu et al., 2024): W' = m * (W + (alpha / r) B A) / ||W + (alpha / r) B A||_c.

    `m` is a per-output-neuron magnitude initialized to the frozen weight's norm and B starts at zero, so the wrapped
    layer is unchanged at init.
    """

    def __init__(self, base: nn.Linear, rank=32, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)  # A
        self.up = nn.Linear(rank, base.out_features, bias=False)  # B
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1))

    def forward(self, x):
        weight = self.base.weight + self.scale * (self.up.weight @ self.down.weight)
        weight = self.magnitude.unsqueeze(1) * weight / (weight.norm(dim=1, keepdim=True) + 1e-6)
        return F.linear(x, weight, self.base.bias)


class ConvAdapter(nn.Module):
    """Convpass (Jie & Deng, 2022): patch tokens += up(GELU(conv3x3(GELU(down(tokens))))) on the 32 x 32 token grid.

    DoRA updates each token on its own; the 3x3 conv gives the frozen ViT the local filtering that speckle and tiny
    objects need. `up` is zero-initialized, so the adapter starts as a no-op. Prefix (CLS / register) tokens pass
    through unchanged.
    """

    def __init__(self, dim=768, hidden=256, num_prefix_tokens=0):
        super().__init__()
        self.prefix = num_prefix_tokens
        self.down = nn.Linear(dim, hidden)
        self.conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens):
        patches = tokens[:, self.prefix:]
        side = int(round(patches.shape[1] ** 0.5))  # square inputs: 512 px -> 32 x 32 tokens
        hidden = F.gelu(self.down(patches)).transpose(1, 2).unflatten(2, (side, side))
        delta = self.up(F.gelu(self.conv(hidden)).flatten(2).transpose(1, 2))
        return torch.cat([tokens[:, :self.prefix], patches + delta], dim=1)


class AdaptedBlock(nn.Module):
    """A transformer block followed by its ConvAdapter: y = x' + adapter(x'), x' = block(x). The block's own
    arguments (RoPE embeddings) are forwarded unchanged."""

    def __init__(self, block, adapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, hidden_states, *args, **kwargs):
        return self.adapter(self.block(hidden_states, *args, **kwargs))


def adapt_backbone(backbone, targets=('attn', 'mlp'), rank=32, alpha=32, convpass_dim=256):
    """Freeze `backbone`, wrap its `targets` projections in DoRA and add a Convpass adapter after every block
    (in place)."""
    backbone.requires_grad_(False)
    for _, parent, name in backbone.adapter_targets(targets):
        setattr(parent, name, DoRA(getattr(parent, name), rank=rank, alpha=alpha))
    for index in range(len(backbone.blocks)):  # after DoRA, so adapter_targets still saw the plain blocks
        adapter = ConvAdapter(backbone.embed_dim, convpass_dim, backbone.num_prefix_tokens)
        backbone.blocks[index] = AdaptedBlock(backbone.blocks[index], adapter)
