"""The DINOv3 backbone, built for 512 px input: images [B, 3, 512, 512] -> patch tokens [B, 1024, embed_dim]
(or [B, 4096, embed_dim] with `patch_size: 8`).

It exposes `embed_dim`, `patch_size`, `num_prefix_tokens`, a `blocks` list, `adapter_targets()` (the projections that
DoRA may wrap, with their block index) and a forward that returns the final patch tokens (prefix tokens dropped).
"""
import torch
import torch.nn.functional as F
from torch import nn


def resize_patch_kernel(weight, size):
    """FlexiViT's PI-resize (Beyer et al., 2023) of a patch-embedding kernel [D, C, p, p] to [D, C, size, size].

    The new kernel on a size x size patch gives the token the old kernel gives on that patch bilinearly resized to
    p x p: <B x, w> = <x, B^T w>, with B the resize matrix. Going from 16 to 8 px, every 8 x 8 patch is embedded
    exactly as its 2x bilinear upsampling would be (about the image upsampled 2x, up to the patch borders), so the
    frozen ViT sees tokens like the ones it was pretrained on (for
    upsampling to a larger patch the pseudo-inverse of B^T would be needed instead; not used here).
    """
    old = weight.shape[-1]
    if size > old:
        raise ValueError(f'patch_size {size} > the pretrained {old}: only smaller patches are supported')
    basis = torch.eye(size * size, dtype=torch.float64).view(-1, 1, size, size)
    resize = F.interpolate(basis, size=(old, old), mode='bilinear', align_corners=False).view(size * size, -1)
    resized = weight.double().flatten(2) @ resize.T  # [D, C, old*old] @ [old*old, size*size]
    return resized.view(*weight.shape[:2], size, size).to(weight.dtype)


class DINOv3(nn.Module):
    """DINOv3 ViT (Meta) through HuggingFace transformers. A single gray channel is repeated to the RGB
    channel count of the pretrained model.

    `model_name` selects the checkpoint (default facebook/dinov3-vitb16-pretrain-lvd1689m, 86 M parameters).
    The checkpoints are gated on Hugging Face: accept the licence on the model page and `huggingface-cli login`
    once before the first run.

    `patch_size` (default: the checkpoint's 16) below 16 re-tokenizes the same 512 px input on a finer grid: 8 gives
    64 x 64 tokens, so half of the boxes are no longer smaller than one token. The patch embedding is PI-resized
    (`resize_patch_kernel`) and RoPE takes its grid from the input and the patch size, so nothing else changes;
    the cost is 4x the tokens (16x the attention).
    """

    def __init__(self, pretrained: bool = True, model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                 grad_checkpointing: bool = False, patch_size: int | None = None, **kwargs):
        super().__init__()
        from transformers import DINOv3ViTConfig, DINOv3ViTModel  # lazy: keeps `import sarbench` light
        if pretrained:
            self.dinov3 = DINOv3ViTModel.from_pretrained(model_name, **kwargs)
        else:
            self.dinov3 = DINOv3ViTModel(DINOv3ViTConfig.from_pretrained(model_name, **kwargs))
        if grad_checkpointing:  # recompute blocks in backward: much lower activation memory, ~30 % slower
            self.dinov3.gradient_checkpointing_enable()
            if hasattr(self.dinov3, 'enable_input_require_grads'):  # frozen embeddings still need a grad path
                self.dinov3.enable_input_require_grads()
        if patch_size is not None and patch_size != self.dinov3.config.patch_size:
            embed = self.dinov3.embeddings.patch_embeddings
            resized = nn.Conv2d(embed.in_channels, embed.out_channels, patch_size, stride=patch_size)
            with torch.no_grad():
                resized.weight.copy_(resize_patch_kernel(embed.weight, patch_size))
                resized.bias.copy_(embed.bias)
            resized.requires_grad_(embed.weight.requires_grad)
            self.dinov3.embeddings.patch_embeddings = resized
            self.dinov3.config.patch_size = patch_size  # the RoPE grid is input size // config.patch_size
        self.model_name = model_name
        self.patch_size = self.dinov3.config.patch_size
        self.embed_dim = self.dinov3.config.hidden_size
        self.in_chans = self.dinov3.config.num_channels
        self.num_prefix_tokens = 1 + getattr(self.dinov3.config, 'num_register_tokens', 0)
        encoder = getattr(self.dinov3, 'model', self.dinov3)  # transformers >= 5 nests the encoder under .model
        self._layers = encoder.layer

    @property
    def blocks(self):
        return self._layers

    def adapter_targets(self, groups=('attn', 'mlp')):
        """(block index, parent module, attribute) of every projection DoRA may wrap. `groups`: 'attn' =
        q_proj/k_proj/v_proj/o_proj, 'mlp' = up_proj/down_proj (gate_proj too when the MLP is gated)."""
        targets = []
        for index, layer in enumerate(self.blocks):
            if 'attn' in groups:
                targets += [(index, layer.attention, name) for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
            if 'mlp' in groups:
                targets += [(index, layer.mlp, name) for name in ('gate_proj', 'up_proj', 'down_proj')
                            if hasattr(layer.mlp, name)]
        return targets

    def unfreeze_patch_embed(self):
        """Re-enable gradients on the patch embedding (adapt_backbone freezes everything; this is called after)."""
        self.dinov3.get_input_embeddings().requires_grad_(True)

    def forward(self, x, with_prefix=False):
        """Patch tokens [B, N, embed_dim]; with `with_prefix`, all tokens [B, prefix + N, embed_dim] (CLS first,
        then the registers)."""
        if x.shape[1] != self.in_chans:  # gray -> the pretrained RGB channel count
            x = x.repeat(1, self.in_chans, 1, 1)
        tokens = self.dinov3(x).last_hidden_state  # [B, prefix + N, embed_dim], final norm applied
        return tokens if with_prefix else tokens[:, self.num_prefix_tokens:]
