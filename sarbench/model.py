"""The multi-task model: one shared ViT backbone feeds a classification head and a Faster R-CNN detector.

Configurable pieces (`model.*` in the YAML): `detail_stem` (real stride-4 detail for P2), `fusion_layers`
(learnable fusion of several ViT blocks before the neck) and `head` (`standard` | `deform` | `cascade` RoI head).
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign

from .adapters import set_task
from .heads import CascadeRoIHeads, DeformConvBoxHead, DetRoIHeads, FocalRPN


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the single stride-16 ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64."""

    def __init__(self, dim=768, channels=256):
        super().__init__()
        self.rescale = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GroupNorm(32, dim // 2), nn.GELU(),
                          nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),  # stride 4
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 8
            nn.Identity(),  # stride 16
            nn.MaxPool2d(2),  # stride 32
        ])
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.GroupNorm(32, channels),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels))
            for c in (dim // 4, dim // 2, dim, dim))

    def forward(self, x):
        levels = [project(rescale(x)) for rescale, project in zip(self.rescale, self.project)]
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class ConvStem(nn.Module):
    """A small conv stem on the raw input image: real stride-4 detail for the (otherwise upsampled) P2 level.

    The single-scale ViT only sees 16 px patches, so the neck's stride-4 level is upsampled and has no genuine
    high-frequency content. The stem reads the 512 px input directly and produces stride-4 features that are
    added to P2, which is what tiny SAR objects need.
    """

    def __init__(self, in_chans=3, channels=64, out_channels=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, channels, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, channels * 2), nn.GELU(),
            nn.Conv2d(channels * 2, out_channels, 1, bias=False), nn.GroupNorm(32, out_channels))

    def forward(self, x):
        return self.stem(x)


class LayerFusion(nn.Module):
    """Learnable softmax-weighted sum of the tokens from several ViT blocks (`model.fusion_layers`).

    Earlier blocks carry lower-level, higher-resolution detail that the final block has abstracted away; fusing
    them before the neck gives the FPN richer features for small objects. Initialized as (almost) the last
    block, so it starts from the standard single-layer behaviour.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))
        with torch.no_grad():
            self.weights[-1] = 3.0

    def forward(self, tokens):
        weights = self.weights.softmax(0).view(-1, 1, 1)
        return sum(weight * token for weight, token in zip(weights, tokens))


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and Faster R-CNN detections."""

    def __init__(self, backbone, task_routing: bool, num_classes=9, detail_stem=False,
                 fusion_layers=None, head='standard', head_params=None, focal_loss=False, focal_gamma=2.0,
                 focal_alpha=0.25, giou_weight=0.0, roi_sampling_ratio=2, roi_output_size=7):
        super().__init__()
        self.backbone = backbone
        self.task_routing = task_routing  # MoE-LoRA: the backbone runs once per task, with that task's routers
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        self.neck = SimpleFeaturePyramid(backbone.embed_dim)
        self.detail_stem = ConvStem(backbone.in_chans) if detail_stem else None  # real detail for P2
        # Detection-loss / RoI options shared by every head.
        self.focal_loss, self.focal_gamma, self.focal_alpha = focal_loss, focal_gamma, focal_alpha
        self.giou_weight = giou_weight
        self.roi_sampling_ratio, self.roi_output_size = roi_sampling_ratio, roi_output_size
        # Multi-layer fusion: normalize negative indices, register hooks in execution order.
        self.fusion_layers = self.fusion = None
        if fusion_layers:
            blocks = len(backbone.blocks)
            self.fusion_layers = sorted(i % blocks for i in fusion_layers)
            self.fusion = LayerFusion(len(self.fusion_layers))
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location.
        anchors = AnchorGenerator(((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), ((0.5, 1.0, 2.0),) * 5)
        rpn_class = FocalRPN if focal_loss else RegionProposalNetwork
        self.rpn = rpn_class(
            anchors, RPNHead(256, 6), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=2000),
            post_nms_top_n=dict(training=2000, testing=2000), nms_thresh=0.7,
            **({'focal_gamma': focal_gamma, 'focal_alpha': focal_alpha} if focal_loss else {}))
        self.roi_heads = self._build_roi_heads(num_classes, head, head_params or {})

    def _build_roi_heads(self, num_classes, head, params):
        """The RoI head ablation: torchvision's head (with focal/GIoU), a Deformable-Conv head, or Cascade.

        `hidden_dim` (default 1024) sets the box-head width for every variant.
        """
        roi_pool = MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=self.roi_output_size,
                                      sampling_ratio=self.roi_sampling_ratio)
        hidden = params.get('hidden_dim', 1024)
        loss_opts = dict(focal_loss=self.focal_loss, focal_gamma=self.focal_gamma, giou_weight=self.giou_weight)
        common = dict(fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
                      bbox_reg_weights=None, score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)
        if head == 'standard':
            return DetRoIHeads(roi_pool, TwoMLPHead(256 * 7 * 7, hidden),
                               FastRCNNPredictor(hidden, num_classes + 1), **common, **loss_opts)
        if head == 'deform':
            keys = ('in_channels', 'kernel_size', 'num_convs')
            return DetRoIHeads(roi_pool, DeformConvBoxHead(out_channels=hidden,
                                                           **{k: params[k] for k in keys if k in params}),
                               FastRCNNPredictor(hidden, num_classes + 1), **common, **loss_opts)
        if head == 'cascade':
            stages = params.get('num_stages', 3)
            # A per-stage head at hidden 1024 costs ~14 M each; the head is shared by default. A smaller
            # hidden_dim lets you afford a full per-stage cascade within the 40 M trainable budget.
            box_heads = [TwoMLPHead(256 * 7 * 7, hidden)
                         for _ in range(1 if params.get('share_head', True) else stages)]
            predictors = [FastRCNNPredictor(hidden, num_classes + 1) for _ in range(stages)]
            thresholds = tuple(params.get('fg_iou_thresholds', ())) \
                or tuple(round(0.5 + 0.1 * i, 2) for i in range(stages))
            return CascadeRoIHeads(
                roi_pool, box_heads, predictors, fg_iou_thresholds=thresholds,
                bg_iou_threshold=params.get('bg_iou_threshold', 0.5),
                batch_size_per_image=512, positive_fraction=0.25, bbox_reg_weights=None,
                score_thresh=0.05, nms_thresh=0.5, detections_per_img=100,
                stage_loss_weights=params.get('stage_loss_weights'), **loss_opts)
        raise ValueError(f"unknown head '{head}'; available: standard, deform, cascade")

    def _backbone_tokens(self, images):
        """Run the backbone; with fusion, return the learnable weighted sum of the selected blocks' tokens."""
        if self.fusion is None:
            return self.backbone(images)
        captured = []
        handles = [self.backbone.blocks[i].register_forward_hook(
            lambda module, args, output: captured.append(output[0] if isinstance(output, tuple) else output))
            for i in self.fusion_layers]
        try:
            self.backbone(images)
        finally:
            for handle in handles:
                handle.remove()
        prefix = getattr(self.backbone, 'num_prefix_tokens', 0)
        return self.fusion([tokens[:, prefix:] for tokens in captured])

    def forward(self, images, targets=None):
        """images [B, C, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the detection losses (training mode))."""
        set_task(self.backbone, 0)  # 0 = classification, 1 = detection; a no-op without MoE-LoRA
        tokens = self._backbone_tokens(images)
        logits = self.cls_head(tokens.mean(dim=1))
        if self.task_routing:
            set_task(self.backbone, 1)
            tokens = self._backbone_tokens(images)
        h, w = images.shape[-2] // 16, images.shape[-1] // 16  # one token per 16 x 16 patch
        features = self.neck(tokens.transpose(1, 2).unflatten(2, (h, w)))  # tokens as a [B, 768, h, w] map
        if self.detail_stem is not None:  # add genuine stride-4 detail (from the pixels) to the upsampled P2
            features['0'] = features['0'] + self.detail_stem(images.to(features['0'].dtype))
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}
