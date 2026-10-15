"""Backbone adaptation: DoRA on the frozen DINOv3, plus the per-block capacity options.

`adapt_backbone` freezes the backbone and then, per block:
    - `full_blocks`: the block trains entirely (no DoRA there; its pretrained weights are unfrozen);
    - every other block: its selected projections (`targets`: attn q/k/v/o, mlp up/down) are wrapped in DoRA, at
      `rank` or the block's entry in `rank_overrides` (alpha / rank is kept, so the update scale is unchanged);
    - `conv_layers`: a Convpass adapter on the block (`conv_dim` wide, `conv_kernel` dense | depthwise | multiscale,
      `conv_placement` sequential | parallel), the only option that mixes neighbouring patches.

Unfreezing an existing block adds trainable parameters only; DoRA and Convpass add new parameters, which count
towards the total budget too.
"""
import torch
import torch.nn.functional as F
from torch import nn


class DoRA(nn.Module):
    """Weight-decomposed LoRA (Liu et al., 2024): W' = m * (W + (alpha / r) B A) / ||W + (alpha / r) B A||_c.

    `m` is a per-output-neuron magnitude initialized to the frozen weight's column norm, and the update is
    renormalized, so init is an exact no-op and training starts from the pretrained layer (as LoRA does).
    """

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)  # A
        self.up = nn.Linear(rank, base.out_features, bias=False)  # B
        nn.init.zeros_(self.up.weight)  # B = 0 -> W' = m * W / ||W||_c = W at init
        self.scale = alpha / rank
        self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1))  # ||W||_c per output neuron

    def forward(self, x):
        weight = self.base.weight + self.scale * (self.up.weight @ self.down.weight)
        weight = self.magnitude.unsqueeze(1) * weight / (weight.norm(dim=1, keepdim=True) + 1e-6)
        return F.linear(x, weight, self.base.bias)


class ConvAdapter(nn.Module):
    """Convpass-style adapter (Jie & Deng, 2022): delta = up(GELU(spatial(GELU(down(tokens))))) on the patch tokens.

    DoRA updates act on each 16 px token independently; the conv over the 32 x 32 token grid gives the backbone the
    local spatial filtering that speckle and tiny objects need. `up` starts at zero, so the adapter is a no-op at
    init. Prefix (CLS / register) tokens are never changed. `kernel` selects the spatial op on the `hidden` channels:

        dense       one 3x3 conv (hidden x hidden x 9 weights: most of the adapter's parameters)
        depthwise   one 3x3 depthwise conv (hidden x 9 weights), so `hidden` can be much wider for the same budget
        multiscale  two 3x3 convs side by side, dilation 1 and 2, each producing hidden / 2 channels (3x3 and 5x5
                    context for the same parameters as `dense`)

    `norm` adds a LayerNorm on the adapter input (used by the parallel placement, which reads the block input).
    """

    def __init__(self, dim=768, hidden=128, num_prefix_tokens=0, kernel='dense', norm=False):
        super().__init__()
        self.prefix = num_prefix_tokens
        self.norm = nn.LayerNorm(dim) if norm else None
        self.down = nn.Linear(dim, hidden)
        if kernel == 'dense':
            self.conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        elif kernel == 'depthwise':
            self.conv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        elif kernel == 'multiscale':
            if hidden % 2:
                raise ValueError('multiscale Convpass needs an even hidden width')
            self.conv = nn.Conv2d(hidden, hidden // 2, 3, padding=1)
            self.conv_dilated = nn.Conv2d(hidden, hidden // 2, 3, padding=2, dilation=2)
        else:
            raise ValueError(f"unknown Convpass kernel '{kernel}'; available: dense, depthwise, multiscale")
        self.kernel = kernel
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def delta(self, tokens):
        """The update for the patch tokens [B, N, dim] (prefix tokens excluded)."""
        patches = tokens[:, self.prefix:]
        if self.norm is not None:
            patches = self.norm(patches)
        side = int(round(patches.shape[1] ** 0.5))  # square inputs: 512 px -> 32 x 32 tokens
        hidden = F.gelu(self.down(patches)).transpose(1, 2).unflatten(2, (side, side))
        if self.kernel == 'multiscale':
            hidden = torch.cat([self.conv(hidden), self.conv_dilated(hidden)], dim=1)
        else:
            hidden = self.conv(hidden)
        return self.up(F.gelu(hidden).flatten(2).transpose(1, 2))

    def forward(self, tokens, source=None):
        """tokens + delta, where the delta is computed from `source` (default: `tokens` themselves)."""
        delta = self.delta(tokens if source is None else source)
        return torch.cat([tokens[:, :self.prefix], tokens[:, self.prefix:] + delta], dim=1)


class AdaptedBlock(nn.Module):
    """A transformer block plus a ConvAdapter; forwards the block's own arguments (RoPE) unchanged.

    `sequential` (default): y = x' + adapter(x') with x' = block(x), the adapter reads the block output.
    `parallel` (Convpass paper): y = block(x) + adapter(x), the adapter reads the block input, side by side with
    attention and MLP.
    """

    def __init__(self, block, adapter, placement='sequential'):
        super().__init__()
        if placement not in ('sequential', 'parallel'):
            raise ValueError(f"unknown Convpass placement '{placement}'; available: sequential, parallel")
        self.block = block
        self.adapter = adapter
        self.placement = placement

    def forward(self, hidden_states, *args, **kwargs):
        out = self.block(hidden_states, *args, **kwargs)
        return self.adapter(out, source=hidden_states if self.placement == 'parallel' else None)


def adapt_backbone(backbone, targets=('attn', 'mlp'), rank=32, alpha=32, rank_overrides=None, full_blocks=(),
                   conv_layers=(), conv_dim=128, conv_kernel='dense', conv_placement='sequential'):
    """Freeze `backbone`, then apply the per-block options described in the module docstring (in place)."""
    blocks = len(backbone.blocks)
    full_blocks = {i % blocks for i in full_blocks}
    conv_layers = sorted({i % blocks for i in conv_layers})
    rank_overrides = {int(i) % blocks: int(r) for i, r in (rank_overrides or {}).items()}
    if full_blocks & set(rank_overrides):
        raise ValueError('a block cannot be both fully trained (full_blocks) and DoRA-adapted (rank_overrides)')
    backbone.requires_grad_(False)
    for index, parent, name in backbone.adapter_targets(targets):
        if index in full_blocks:
            continue
        block_rank = rank_overrides.get(index, rank)
        setattr(parent, name, DoRA(getattr(parent, name), rank=block_rank, alpha=alpha * block_rank / rank))
    for index in full_blocks:
        backbone.blocks[index].requires_grad_(True)
    for index in conv_layers:  # after DoRA, so adapter_targets still sees the plain blocks
        adapter = ConvAdapter(backbone.embed_dim, conv_dim, backbone.num_prefix_tokens, kernel=conv_kernel,
                              norm=conv_placement == 'parallel')
        backbone.blocks[index] = AdaptedBlock(backbone.blocks[index], adapter, placement=conv_placement)
