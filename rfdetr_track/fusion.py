"""DINOv3 feature fusion, an image-classification head and a P3 level for RF-DETR, added without editing rfdetr.

`install(spec)` patches rfdetr's model builders, so every RF-DETR model built afterwards (the one train_rfdetr.py
counts, the one rfdetr trains, the one predict_rfdetr.py loads a checkpoint into) gets:

  - `dino`: a frozen DINOv3 ViT (sarbench.backbones.DINOv3, the backbone of the Faster R-CNN track) on the same
    normalised 512 px input. Its 32 x 32 patch tokens (stride 16) go through a LayerNorm and, per projector level, a
    zero-initialised 1 x 1 conv into the decoder width, and are added to RF-DETR's own projector output (resized to
    the level's grid; P4 is the same 32 x 32 grid). At init the model is exactly pretrained RF-DETR.
    `dora_rank` / `convpass_dim` adapt the frozen ViT with sarbench's DoRA and Convpass (adapt_backbone), as in
    the Faster R-CNN track; without them it runs under no_grad.
  - `image_cls_weight` > 0: a linear image classifier on DINOv3's [CLS, mean patch] tokens, trained jointly with a
    cross-entropy term (`loss_image_cls`, weight `image_cls_weight`) whose target is the class of the image's boxes
    (every course image holds one class). Its logits come out as outputs['pred_image_logits']; predict_rfdetr.py
    takes the image label from them.

and, independently of the spec, pretrained checkpoints with a P4-only projector load into a P3 + P4 model
(`projector_scale: [P3, P4]`, train_rfdetr.py --p3): the pretrained P4 stage moves to its new index, the new P3
stage takes every pretrained P4 weight of a matching shape, and each decoder deformable-attention layer keeps its
pretrained P4 sampling offsets / attention weights, copied to P3 (offsets are relative to the reference box, so they
mean the same at both levels) with the P3 attention logits lowered by P3_ATTENTION_OFFSET so the decoder starts
reading mostly P4, as pretrained.

The new parameters live under backbone.0.dino* / backbone.0.image_head: rfdetr's optimizer puts them in its default
group (lr), and its checkpoints save them, so `install` with the same spec before loading restores them.
"""
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rfdetr.models.backbone.backbone import Backbone  # noqa: E402
from rfdetr.models.criterion import SetCriterion  # noqa: E402
from rfdetr.models.lwdetr import LWDETR  # noqa: E402
from rfdetr.utilities.tensors import NestedTensor  # noqa: E402

SPEC_FILE = 'fusion.json'
NUM_CLASSES = 9
P3_ATTENTION_OFFSET = -4.0  # P3's share of each head's attention starts at ~e^-4 of P4's
DEFAULT_SPEC = {'dino': None, 'image_cls_weight': 0.0}
DEFAULT_DINO = {'model_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m', 'pretrained': True,  # False: tests
                'dora_rank': 0, 'dora_alpha': None,
                'dora_targets': ['attn', 'mlp'], 'convpass_dim': 0, 'convpass_layers': None}

_spec = None
_original = {}


def full_spec(spec):
    """`spec` with every default filled in (dino None = no DINOv3 branch)."""
    spec = {**DEFAULT_SPEC, **(spec or {})}
    if spec['dino'] is not None:
        spec['dino'] = {**DEFAULT_DINO, **spec['dino']}
    if spec['image_cls_weight'] and spec['dino'] is None:
        raise ValueError('the image classifier reads DINOv3 tokens: image_cls_weight needs a dino branch')
    return spec


def save_spec(spec, directory):
    Path(directory, SPEC_FILE).write_text(json.dumps(full_spec(spec), indent=1))


def load_spec(directory):
    """The spec saved next to a run's checkpoints, or None for a plain RF-DETR run."""
    path = Path(directory, SPEC_FILE)
    return json.loads(path.read_text()) if path.exists() else None


class FusedBackbone(Backbone):
    """rfdetr's Backbone (encoder + projector) plus the DINOv3 branch added by `attach`."""

    def forward(self, tensor_list):
        out, cross_attn_out = super().forward(tensor_list)
        images = tensor_list.tensors
        trainable = any(p.requires_grad for p in self.dino.parameters())
        with nullcontext() if trainable else torch.no_grad():
            tokens = self.dino(images, with_prefix=True)  # [B, prefix + N, D], DINOv3's final norm applied
        prefix = self.dino.num_prefix_tokens
        patch = self.dino.patch_size
        grid = self.dino_norm(tokens[:, prefix:]).transpose(1, 2)
        grid = grid.unflatten(2, (images.shape[-2] // patch, images.shape[-1] // patch))  # [B, D, H/16, W/16]
        fused = []
        for level, conv in zip(out, self.dino_fuse):
            feat = level.tensors
            source = grid if grid.shape[-2:] == feat.shape[-2:] else F.interpolate(
                grid, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            fused.append(NestedTensor(feat + conv(source).to(feat.dtype), level.mask, level.no_padding))
        self._image_logits = None
        if self.image_head is not None:
            pooled = torch.cat([tokens[:, 0], tokens[:, prefix:].mean(dim=1)], dim=1)
            self._image_logits = self.image_head(pooled.float())
        return fused, cross_attn_out


class FusedLWDETR(LWDETR):
    """LWDETR whose outputs also carry the image-classifier logits (when the backbone has a classifier)."""

    def forward(self, samples, targets=None):
        out = super().forward(samples, targets)
        logits = getattr(self.backbone[0], '_image_logits', None)
        if logits is not None:
            out['pred_image_logits'] = logits
        return out


class ImageClsCriterion(SetCriterion):
    """SetCriterion plus the image-classification cross-entropy (`loss_image_cls`)."""

    def forward(self, outputs, targets, num_boxes=None):
        losses = super().forward(outputs, targets, num_boxes)
        logits = outputs.get('pred_image_logits')
        if logits is not None:
            losses['loss_image_cls'] = image_cls_loss(logits, targets)
        return losses


def image_labels(targets):
    """Per image, the most common box class (0-8), or -1 for an image left without boxes (e.g. by a crop)."""
    labels = []
    for target in targets:
        boxes = target['labels']
        labels.append(int(torch.bincount(boxes).argmax()) if boxes.numel() else -1)
    return torch.tensor(labels, dtype=torch.long)


def image_cls_loss(logits, targets):
    labels = image_labels(targets).to(logits.device)
    if (labels < 0).all():
        return logits.sum() * 0.0
    return F.cross_entropy(logits.float(), labels, ignore_index=-1)


def attach(model, spec):
    """Add the spec's DINOv3 branch and image classifier to a freshly built LWDETR (in place)."""
    dino_spec = spec['dino']
    if dino_spec is None:
        return model
    from sarbench.adapters import adapt_backbone
    from sarbench.backbones import DINOv3
    backbone = model.backbone[0]
    dino = DINOv3(pretrained=dino_spec['pretrained'], model_name=dino_spec['model_name'])
    dino.requires_grad_(False)
    if dino_spec['dora_rank'] or dino_spec['convpass_dim']:
        rank = dino_spec['dora_rank']
        blocks = len(dino.blocks)
        adapt_backbone(dino, targets=tuple(dino_spec['dora_targets']) if rank else (), rank=rank or 1,
                       alpha=dino_spec['dora_alpha'] or rank or 1,
                       conv_layers=(dino_spec['convpass_layers'] if dino_spec['convpass_layers'] is not None
                                    else range(blocks)) if dino_spec['convpass_dim'] else (),
                       conv_dim=dino_spec['convpass_dim'] or 128)
    width = model.transformer.d_model
    backbone.dino = dino
    backbone.dino_norm = nn.LayerNorm(dino.embed_dim)
    backbone.dino_fuse = nn.ModuleList(nn.Conv2d(dino.embed_dim, width, 1) for _ in backbone.projector_scale)
    for conv in backbone.dino_fuse:
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
    backbone.image_head = nn.Linear(2 * dino.embed_dim, NUM_CLASSES) if spec['image_cls_weight'] else None
    backbone._image_logits = None
    if type(backbone) is not Backbone or type(model) is not LWDETR:
        raise TypeError(f'expected rfdetr Backbone / LWDETR, got {type(backbone).__name__} / {type(model).__name__}')
    backbone.__class__ = FusedBackbone
    model.__class__ = FusedLWDETR
    return model


def remap_p3(model, state):
    """A P4-only checkpoint's weights, laid out for a P3 + P4 model (see the module docstring); others unchanged."""
    keys = model.state_dict()
    old, new = 'backbone.0.projector.stages.0.', 'backbone.0.projector.stages.1.'
    if not any(k.startswith(new) for k in keys) or any(k.startswith(new) for k in state):
        return state
    state = dict(state)
    p4 = {k[len(old):]: v for k, v in state.items() if k.startswith(old)}
    for suffix, value in p4.items():
        state[new + suffix] = value  # the pretrained P4 stage, at P4's index in the P3 + P4 projector
        if keys.get(old + suffix) is None or keys[old + suffix].shape != value.shape:
            del state[old + suffix]  # the new P3 stage keeps its init where shapes differ (its input conv)
    for name, module in model.named_modules():
        if not (hasattr(module, 'sampling_offsets') and hasattr(module, 'n_levels')) or module.n_levels != 2:
            continue
        heads, points = module.n_heads, module.n_points
        for part, tail in (('sampling_offsets', (points, 2)), ('attention_weights', (points,))):
            for kind in ('weight', 'bias'):
                key = f'{name}.{part}.{kind}'
                if key not in state or state[key].shape == keys[key].shape:
                    continue
                pretrained = state[key].reshape(heads, 1, *tail, *state[key].shape[1:])  # one level: P4
                p3 = pretrained.clone()
                if part == 'attention_weights' and kind == 'bias':
                    p3 = p3 + P3_ATTENTION_OFFSET
                state[key] = torch.cat([p3, pretrained], dim=1).reshape(keys[key].shape)  # level order P3, P4
    return state


def _build_model(args):
    model = _original['build_model'](args)
    if _spec is not None and isinstance(model, LWDETR):
        attach(model, _spec)
    return model


def _build_criterion(*args, **kwargs):
    criterion, postprocess = _original['build_criterion'](*args, **kwargs)
    if _spec is not None and _spec['image_cls_weight']:
        if type(criterion) is not SetCriterion:
            raise TypeError(f'expected rfdetr SetCriterion, got {type(criterion).__name__}')
        criterion.__class__ = ImageClsCriterion
        criterion.weight_dict['loss_image_cls'] = float(_spec['image_cls_weight'])
    return criterion, postprocess


def _load_model_state(nn_model, model_state, **kwargs):
    return _original['load_model_state'](nn_model, remap_p3(nn_model, model_state), **kwargs)


def install(spec=None):
    """Patch rfdetr so models built from now on follow `spec` (None: plain RF-DETR, but P3 loading still works)."""
    global _spec
    import rfdetr.inference
    import rfdetr.models.lwdetr
    import rfdetr.models.weights
    import rfdetr.training.module_model
    if not _original:
        _original['build_model'] = rfdetr.models.lwdetr.build_model
        _original['build_criterion'] = rfdetr.training.module_model.build_criterion_from_config
        _original['load_model_state'] = rfdetr.models.weights._load_model_state
        rfdetr.models.lwdetr.build_model = _build_model  # build_model_from_config (training) calls this
        rfdetr.inference.build_model = _build_model  # RFDETRLarge() / from_checkpoint (inference) call this
        rfdetr.training.module_model.build_criterion_from_config = _build_criterion
        rfdetr.models.weights._load_model_state = _load_model_state
    _spec = full_spec(spec) if spec is not None else None
    return _spec
