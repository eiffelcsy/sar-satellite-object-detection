"""The DINOv3 ViT-B/16 backbone (Hugging Face transformers): images [B, 3, 512, 512] -> patch tokens [B, 1024, 768].

It exposes `embed_dim`, `patch_size`, `in_chans`, `num_prefix_tokens`, a `blocks` list and `adapter_targets()` (the
projections DoRA wraps), and its forward drops the CLS and 4 register tokens.
"""
from pathlib import Path

from torch import nn

# The checkpoint's architecture (its config.json), so inference can build the model without the gated download:
# every weight then comes from the trained model.pt.
CONFIG_JSON = Path(__file__).with_name('dinov3_vitb16_config.json')


class DINOv3(nn.Module):
    """DINOv3 ViT through transformers. `pretrained` loads the (gated) checkpoint `model_name`; otherwise the same
    architecture is built from `CONFIG_JSON` with random weights, to be overwritten by a saved state_dict."""

    def __init__(self, model_name='facebook/dinov3-vitb16-pretrain-lvd1689m', pretrained=True):
        super().__init__()
        from transformers import DINOv3ViTConfig, DINOv3ViTModel  # lazy: keeps `import sarbench` light
        if pretrained:
            self.dinov3 = DINOv3ViTModel.from_pretrained(model_name)
        else:
            self.dinov3 = DINOv3ViTModel(DINOv3ViTConfig.from_json_file(CONFIG_JSON))
        self.patch_size = self.dinov3.config.patch_size
        self.embed_dim = self.dinov3.config.hidden_size
        self.in_chans = self.dinov3.config.num_channels
        self.num_prefix_tokens = 1 + self.dinov3.config.num_register_tokens
        encoder = getattr(self.dinov3, 'model', self.dinov3)  # transformers >= 5 nests the encoder under .model
        self._layers = encoder.layer

    @property
    def blocks(self):
        return self._layers

    def adapter_targets(self, groups=('attn', 'mlp')):
        """(block index, parent module, attribute) of every projection DoRA wraps: 'attn' = q/k/v/o_proj,
        'mlp' = up/down_proj."""
        targets = []
        for index, layer in enumerate(self.blocks):
            if 'attn' in groups:
                targets += [(index, layer.attention, name) for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
            if 'mlp' in groups:
                targets += [(index, layer.mlp, name) for name in ('gate_proj', 'up_proj', 'down_proj')
                            if hasattr(layer.mlp, name)]
        return targets

    def unfreeze_patch_embed(self):
        """Re-enable gradients on the patch embedding (adapt_backbone froze everything)."""
        self.dinov3.get_input_embeddings().requires_grad_(True)

    def forward(self, x):
        """Patch tokens [B, N, embed_dim] (final norm applied, prefix tokens dropped)."""
        if x.shape[1] != self.in_chans:  # gray -> the pretrained RGB channel count
            x = x.repeat(1, self.in_chans, 1, 1)
        return self.dinov3(x).last_hidden_state[:, self.num_prefix_tokens:]
