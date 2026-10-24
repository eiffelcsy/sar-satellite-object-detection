"""ViT-Adapter (Chen et al., ICLR 2023): a convolutional spatial prior that talks to the frozen ViT in both directions.

The ViT only sees 16 px patches; the P2 conv stem and Convpass add local detail after the backbone or between
neighbouring tokens. ViT-Adapter instead runs a small CNN (the spatial prior module, SPM) on the full-resolution
image, at strides 4, 8, 16 and 32, and at N interaction points:

    injector i   before block start_i: the ViT patch tokens cross-attend to the SPM features at strides 8/16/32
                 (deformable attention) and add the result, scaled by a zero-initialized gamma (a no-op at init);
    extractor i  after block end_i: the SPM features cross-attend to the ViT patch tokens and pass a ConvFFN,
                 so the multi-scale features carry the ViT's semantics.

After the backbone, the SPM features (stride 4 plus the extracted strides 8/16/32) are projected to the neck width
and added to the pyramid levels P2-P5, as the P2 stem does for P2 alone.

Deformable attention is written with `F.grid_sample` (as Deformable DETR's PyTorch reference), so there is no CUDA
extension to build. The adapter lives outside the backbone (`model.vit_adapter`, trained at `train.lr`); the blocks
it touches are wrapped in `InteractionBlock`s that hold a plain (unregistered) reference to it.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def grid_centers(h, w, device=None):
    """The (x, y) centres of an h x w grid in [0, 1], row-major (the order of flattened tokens): [h * w, 2]."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    y, x = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([x.flatten(), y.flatten()], dim=1)


class DeformableAttention(nn.Module):
    """Multi-scale deformable attention (Zhu et al., 2021): each query samples `points` locations per head and per
    level around its reference point, at offsets and with weights predicted from the query itself."""

    def __init__(self, query_dim, value_dim, dim, out_dim, heads=6, levels=1, points=4):
        super().__init__()
        if dim % heads:
            raise ValueError(f'deformable attention width {dim} is not divisible by {heads} heads')
        self.heads, self.levels, self.points = heads, levels, points
        self.sampling_offsets = nn.Linear(query_dim, heads * levels * points * 2)
        self.attention_weights = nn.Linear(query_dim, heads * levels * points)
        self.value_proj = nn.Linear(value_dim, dim)
        self.output_proj = nn.Linear(dim, out_dim)
        # Deformable DETR's init: offsets start on a ring of directions (one per head, growing with the point
        # index), uniform attention weights.
        nn.init.zeros_(self.sampling_offsets.weight)
        angles = torch.arange(heads, dtype=torch.float32) * (2 * math.pi / heads)
        directions = torch.stack([angles.cos(), angles.sin()], -1)
        directions = directions / directions.abs().max(-1, keepdim=True).values
        directions = directions.view(heads, 1, 1, 2).repeat(1, levels, points, 1)
        directions *= torch.arange(1, points + 1, dtype=torch.float32).view(1, 1, -1, 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(directions.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        for layer in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, query, reference, value, shapes):
        """query [B, Q, query_dim], reference [Q, 2] ((x, y) in [0, 1]), value [B, sum(h * w), value_dim] (the levels
        flattened and concatenated), shapes [(h, w)] per level -> [B, Q, out_dim]."""
        batch, queries, _ = query.shape
        heads, levels, points = self.heads, self.levels, self.points
        value = self.value_proj(value)
        head_dim = value.shape[-1] // heads
        # Sampling locations in fp32: under bf16 a [0, 1] coordinate would snap to steps of up to 1/256.
        offsets = self.sampling_offsets(query).float().view(batch, queries, heads, levels, points, 2)
        weights = self.attention_weights(query).float().view(batch, queries, heads, levels * points)
        weights = weights.softmax(-1).view(batch, queries, heads, levels, points)
        size = torch.tensor([[w, h] for h, w in shapes], dtype=torch.float32, device=query.device)
        locations = reference.view(1, queries, 1, 1, 1, 2) + offsets / size.view(1, 1, 1, levels, 1, 2)
        grids = 2 * locations - 1  # grid_sample's [-1, 1] (align_corners=False: pixel centres as above)
        out = 0
        for level, ((h, w), level_value) in enumerate(zip(shapes, value.split([h * w for h, w in shapes], dim=1))):
            level_value = level_value.float().view(batch, h * w, heads, head_dim)
            level_value = level_value.permute(0, 2, 3, 1).reshape(batch * heads, head_dim, h, w)
            grid = grids[:, :, :, level].permute(0, 2, 1, 3, 4).reshape(batch * heads, queries, points, 2)
            sampled = F.grid_sample(level_value, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            weight = weights[:, :, :, level].permute(0, 2, 1, 3).reshape(batch * heads, 1, queries, points)
            out = out + (sampled * weight).sum(-1)  # [B * heads, head_dim, Q]
        out = out.view(batch, heads * head_dim, queries).transpose(1, 2)
        return self.output_proj(out.to(query.dtype))


class SpatialPrior(nn.Module):
    """The SPM: a ResNet-like stem (stride 4) and three stride-2 convs -> [c1, c2, c3, c4] at strides 4/8/16/32,
    each projected to `dim` channels."""

    def __init__(self, in_chans=3, channels=64, dim=192):
        super().__init__()

        def conv(cin, cout, stride):
            return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(8, cout), nn.GELU()]

        widths = (channels, 2 * channels, 4 * channels, 4 * channels)
        self.stem = nn.Sequential(*conv(in_chans, channels, 2), *conv(channels, channels, 1),
                                  *conv(channels, channels, 1), nn.MaxPool2d(3, stride=2, padding=1))
        self.downs = nn.ModuleList(nn.Sequential(*conv(cin, cout, 2)) for cin, cout in zip(widths, widths[1:]))
        self.projs = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in widths)

    def forward(self, x):
        features = [self.stem(x)]
        for down in self.downs:
            features.append(down(features[-1]))
        return [proj(f) for proj, f in zip(self.projs, features)]


class ConvFFN(nn.Module):
    """fc -> 3x3 depthwise conv on each level's map -> GELU -> fc, on the flattened multi-level SPM tokens."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)

    def forward(self, x, shapes):
        x = self.fc1(x)
        levels = [self.dwconv(level.transpose(1, 2).unflatten(2, (h, w))).flatten(2).transpose(1, 2)
                  for (h, w), level in zip(shapes, x.split([h * w for h, w in shapes], dim=1))]
        return self.fc2(F.gelu(torch.cat(levels, dim=1)))


class Injector(nn.Module):
    """ViT patch tokens += gamma * DeformAttn(tokens -> SPM strides 8/16/32); gamma starts at 0 (a no-op)."""

    def __init__(self, embed_dim, dim, heads, points):
        super().__init__()
        self.query_norm, self.feat_norm = nn.LayerNorm(embed_dim), nn.LayerNorm(dim)
        self.attn = DeformableAttention(embed_dim, dim, dim, embed_dim, heads, levels=3, points=points)
        self.gamma = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, patches, reference, feats, shapes):
        return patches + self.gamma * self.attn(self.query_norm(patches), reference, self.feat_norm(feats), shapes)


class Extractor(nn.Module):
    """SPM tokens += DeformAttn(SPM tokens -> ViT patch tokens), then += ConvFFN."""

    def __init__(self, embed_dim, dim, heads, points, ffn_ratio):
        super().__init__()
        self.query_norm, self.feat_norm = nn.LayerNorm(dim), nn.LayerNorm(embed_dim)
        self.attn = DeformableAttention(dim, embed_dim, dim, dim, heads, levels=1, points=points)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = ConvFFN(dim, max(1, int(dim * ffn_ratio)))

    def forward(self, feats, reference, patches, grid, shapes):
        feats = feats + self.attn(self.query_norm(feats), reference, self.feat_norm(patches), [grid])
        return feats + self.ffn(self.ffn_norm(feats), shapes)


class InteractionBlock(nn.Module):
    """A backbone block with the adapter's injectors run before it and its extractors after it (by index)."""

    def __init__(self, block, adapter, inject=(), extract=()):
        super().__init__()
        self.block = block
        object.__setattr__(self, '_adapter', adapter)  # not a submodule: the model owns (and counts) the adapter
        self.inject, self.extract = tuple(inject), tuple(extract)

    def forward(self, hidden_states, *args, **kwargs):
        for i in self.inject:
            hidden_states = self._adapter.inject(i, hidden_states)
        out = self.block(hidden_states, *args, **kwargs)
        for i in self.extract:
            self._adapter.extract(i, out[0] if isinstance(out, tuple) else out)
        return out


class ViTAdapter(nn.Module):
    """The SPM, one injector and one extractor per `interactions` entry [start, end] (block indices), and the
    projections of [c1, c2, c3, c4] to the neck width. Call `prepare(images)` before the backbone forward and
    `outputs()` after it."""

    def __init__(self, embed_dim, num_prefix_tokens, patch_size=16, interactions=((0, 2), (3, 5), (6, 8), (9, 11)),
                 in_chans=3, dim=192, heads=6, points=4, ffn_ratio=0.25, channels=64, out_channels=256):
        super().__init__()
        self.prefix, self.patch_size = num_prefix_tokens, patch_size
        self.interactions = [(int(start), int(end)) for start, end in interactions]
        if any(start > end for start, end in self.interactions):
            raise ValueError(f'vit_adapter interactions must be [start, end] with start <= end: {interactions}')
        self.spatial_prior = SpatialPrior(in_chans, channels, dim)
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        nn.init.normal_(self.level_embed, std=0.02)
        self.injectors = nn.ModuleList(Injector(embed_dim, dim, heads, points) for _ in self.interactions)
        self.extractors = nn.ModuleList(Extractor(embed_dim, dim, heads, points, ffn_ratio)
                                        for _ in self.interactions)
        self.out_projs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(dim, out_channels, 1, bias=False), nn.GroupNorm(32, out_channels))
            for _ in range(4))
        self._state = None

    def attach(self, backbone):
        """Wrap the blocks named in `interactions` (after DoRA / Convpass, which wrap them first)."""
        blocks = len(backbone.blocks)
        inject, extract = {}, {}
        for i, (start, end) in enumerate(self.interactions):
            inject.setdefault(start % blocks, []).append(i)
            extract.setdefault(end % blocks, []).append(i)
        for index in sorted(set(inject) | set(extract)):
            backbone.blocks[index] = InteractionBlock(backbone.blocks[index], self, inject.get(index, ()),
                                                      extract.get(index, ()))

    def prepare(self, images):
        c1, *levels = self.spatial_prior(images)
        shapes = [tuple(level.shape[-2:]) for level in levels]
        grid = (images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size)
        feats = torch.cat([level.flatten(2).transpose(1, 2) + embed
                           for level, embed in zip(levels, self.level_embed)], dim=1)
        self._state = {'c1': c1, 'feats': feats, 'shapes': shapes, 'grid': grid,
                       'token_reference': grid_centers(*grid, device=images.device),
                       'feat_reference': torch.cat([grid_centers(h, w, images.device) for h, w in shapes])}

    def _require_state(self):
        if self._state is None:
            raise RuntimeError('ViTAdapter.prepare(images) must run before the backbone forward')
        return self._state

    def inject(self, i, hidden_states):
        state = self._require_state()
        patches = self.injectors[i](hidden_states[:, self.prefix:], state['token_reference'], state['feats'],
                                    state['shapes'])
        return torch.cat([hidden_states[:, :self.prefix], patches.to(hidden_states.dtype)], dim=1)

    def extract(self, i, hidden_states):
        state = self._require_state()
        state['feats'] = self.extractors[i](state['feats'], state['feat_reference'], hidden_states[:, self.prefix:],
                                            state['grid'], state['shapes'])

    def outputs(self):
        """[P2, P3, P4, P5] additions at the neck width; clears the per-forward state."""
        state, self._state = self._require_state(), None
        levels = [f.transpose(1, 2).unflatten(2, shape)
                  for f, shape in zip(state['feats'].split([h * w for h, w in state['shapes']], dim=1),
                                      state['shapes'])]
        c1 = state['c1'] + F.interpolate(levels[0], size=state['c1'].shape[-2:], mode='bilinear',
                                         align_corners=False)
        return [proj(c) for proj, c in zip(self.out_projs, [c1, *levels])]
