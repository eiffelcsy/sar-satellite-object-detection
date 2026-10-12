"""The DINOv3 backbone, built for 512 px input: images [B, 3, 512, 512] -> patch tokens [B, 1024, embed_dim].

It exposes `embed_dim`, `num_prefix_tokens`, a `blocks` list, `adapter_targets()` (the projections that DoRA may
wrap, with their block index) and a forward that returns the final patch tokens (prefix tokens dropped).
"""
from torch import nn


class DINOv3(nn.Module):
    """DINOv3 ViT (Meta) through HuggingFace transformers. A single gray channel is repeated to the RGB
    channel count of the pretrained model.

    `model_name` selects the checkpoint (default facebook/dinov3-vitb16-pretrain-lvd1689m, 86 M parameters).
    The checkpoints are gated on Hugging Face: accept the licence on the model page and `huggingface-cli login`
    once before the first run.
    """

    def __init__(self, pretrained: bool = True, model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                 grad_checkpointing: bool = False, **kwargs):
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
        self.model_name = model_name
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

    def forward(self, x):
        if x.shape[1] != self.in_chans:  # gray -> the pretrained RGB channel count
            x = x.repeat(1, self.in_chans, 1, 1)
        tokens = self.dinov3(x).last_hidden_state  # [B, prefix + N, embed_dim], final norm applied
        return tokens[:, self.num_prefix_tokens:]
