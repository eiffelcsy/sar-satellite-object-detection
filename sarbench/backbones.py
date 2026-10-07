"""Backbones behind a small registry, built for 512 px input: images [B, 1, 512, 512] -> patch tokens [B, N, embed_dim].

Every backbone exposes `embed_dim`, a `blocks` iterable, `adapter_targets()` (the attention projections that a
PEFT method may wrap) and a forward that returns the final patch tokens (prefix/CLS tokens dropped). New backbones
are added with `@register_backbone('name')` and selected from the YAML configs by `backbone.name`.
"""
import timm
from terratorch.registry import BACKBONE_REGISTRY
from torch import nn

BACKBONES = {}


def register_backbone(name):
    """Register a backbone class under `name`, for build_backbone() and the YAML configs."""

    def decorator(cls):
        BACKBONES[name] = cls
        return cls

    return decorator


@register_backbone('vit')
class ViT(nn.Module):
    """timm ViT-B/16 pretrained on ImageNet-21k."""

    def __init__(self, pretrained: bool, timm_name: str = 'vit_base_patch16_224.augreg_in21k',
                 img_size: int = 512, in_chans: int = 1):
        super().__init__()
        # in_chans=1: timm sums the RGB patch-embedding weights; img_size=512: it resamples the position embedding
        self.vit = timm.create_model(timm_name, pretrained=pretrained, img_size=img_size,
                                     in_chans=in_chans, num_classes=0)
        self.embed_dim = self.vit.embed_dim
        self.in_chans = in_chans

    @property
    def blocks(self):
        return self.vit.blocks

    def adapter_targets(self, groups=('attn',)):
        if 'attn' not in groups:
            return []
        return [(blk.attn, name) for blk in self.blocks for name in ('qkv', 'proj')]

    def forward(self, x):
        tokens = self.vit.forward_features(x)  # [B, 1 + N, 768], final norm applied
        return tokens[:, self.vit.num_prefix_tokens:]  # drop the CLS token


@register_backbone('terramind')
class TerraMind(nn.Module):
    """TerraMind-1.0-base (IBM/ESA) through its Sentinel-1 GRD input."""

    def __init__(self, pretrained: bool):
        super().__init__()
        self.terramind = BACKBONE_REGISTRY.build('terratorch_terramind_v1_base', pretrained=pretrained,
                                                 modalities=['S1GRD'])
        self.embed_dim = 768
        self.in_chans = 1  # a single gray channel, repeated to VV/VH inside forward

    @property
    def blocks(self):
        return self.terramind.encoder

    def adapter_targets(self, groups=('attn',)):
        if 'attn' not in groups:
            return []
        return [(blk.attn, name) for blk in self.blocks for name in ('qkv', 'proj')]

    def forward(self, x):
        per_block = self.terramind({'S1GRD': x.repeat(1, 2, 1, 1)})  # gray image as both VV and VH
        return per_block[-1]  # last block's tokens [B, N, 768]; terratorch already applied encoder_norm


@register_backbone('dinov3')
class DINOv3(nn.Module):
    """DINOv3 ViT (Meta) through HuggingFace transformers. A single gray channel is repeated to the RGB
    channel count of the pretrained model.

    `model_name` selects the checkpoint, e.g. facebook/dinov3-vitb16-pretrain-lvd1689m (natural images,
    86 M parameters, within the 130 M budget) or facebook/dinov3-vitl16-pretrain-sat493m (satellite
    imagery, 303 M parameters, over the budget). The DINOv3 checkpoints are gated on Hugging Face: accept
    the licence on the model page and `huggingface-cli login` once before the first run.
    """

    def __init__(self, pretrained: bool, model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                 grad_checkpointing: bool = False, **kwargs):
        super().__init__()
        from transformers import DINOv3ViTConfig, DINOv3ViTModel  # lazy: only DINOv3 needs transformers
        if pretrained:
            self.dinov3 = DINOv3ViTModel.from_pretrained(model_name, **kwargs)
        else:
            self.dinov3 = DINOv3ViTModel(DINOv3ViTConfig.from_pretrained(model_name, **kwargs))
        if grad_checkpointing:  # recompute blocks in backward: much lower activation memory, ~30 % slower
            self.dinov3.gradient_checkpointing_enable()
            if hasattr(self.dinov3, 'enable_input_require_grads'):  # frozen embeddings still need a grad path
                self.dinov3.enable_input_require_grads()
        self.model_name = model_name
        self.embed_dim = self.dinov3.config.hidden_size
        self.in_chans = self.dinov3.config.num_channels
        self.num_prefix_tokens = 1 + getattr(self.dinov3.config, 'num_register_tokens', 0)
        encoder = getattr(self.dinov3, 'model', self.dinov3)  # transformers >= 5 nests the encoder under .model
        self._layers = encoder.layer

    @property
    def blocks(self):
        return self._layers

    def adapter_targets(self, groups=('attn',)):
        """`groups` selects where adapters go: 'attn' = q_proj/k_proj/v_proj/o_proj ('qkv_proj'/'out_proj'),
        'mlp' = up_proj/down_proj (fc1/fc2; gate_proj/up_proj/down_proj when the MLP is gated)."""
        targets = []
        for layer in self.blocks:
            if 'attn' in groups:
                targets += [(layer.attention, name) for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
            if 'mlp' in groups:
                targets += [(layer.mlp, name) for name in ('gate_proj', 'up_proj', 'down_proj')
                            if hasattr(layer.mlp, name)]
        return targets

    def forward(self, x):
        if x.shape[1] != self.in_chans:  # gray -> the pretrained RGB channel count
            x = x.repeat(1, self.in_chans, 1, 1)
        tokens = self.dinov3(x).last_hidden_state  # [B, prefix + N, embed_dim], final norm applied
        return tokens[:, self.num_prefix_tokens:]


def build_backbone(name, pretrained, **params):
    """Build a registered backbone by name; `params` are forwarded to its constructor."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone '{name}'; available: {sorted(BACKBONES)}")
    return BACKBONES[name](pretrained, **params)
