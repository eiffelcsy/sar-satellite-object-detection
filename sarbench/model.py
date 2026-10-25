"""The multi-task model: one shared ViT backbone feeds a classification head and a Faster R-CNN detector.

Configurable pieces (`model.*` in the YAML): `detail_stem` (real stride-4 detail for P2), `fusion_layers`
(learnable fusion of several ViT blocks before the neck), `input_adapter` (ImageNet norm + learnable 1x1 conv +
unfrozen patch embedding), `wst_stem` (wavelet scattering detail for P2), `vit_adapter` (ViT-Adapter's spatial prior
with injectors / extractors, added to P2-P5), `giou_weight` (GIoU box term), `head` (`standard` | `cascade` RoI head),
`box_similarity` (IoU, NWD or their max for anchor / proposal labels), `rescore` (re-score the boxes with the
image classifier at inference), `iou_aware` (an IoU branch; rank boxes by class score x predicted IoU),
`heatmap_aux` (a training-only center-heatmap loss on P2) and `cls_detach` (no classification gradient into the
backbone).
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

from .adapters import adapt_backbone
from .backbones import DINOv3
from .heads import RESCORE_MODES, CascadeRoIHeads, ConvFCBoxHead, DetRoIHeads, box_similarity as make_similarity
from .vit_adapter import ViTAdapter

# Competition limits (Codabench terms): every parameter counts towards the total, frozen ones included.
MAX_TOTAL_PARAMETERS = 130_000_000
MAX_TRAINABLE_PARAMETERS = 40_000_000


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the single ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64.

    `patch_size` is the ViT map's stride: 16 (the default) or 8 (`backbone.params.patch_size: 8`).
    """

    def __init__(self, dim=768, channels=256, patch_size=16):
        super().__init__()
        if patch_size == 16:
            self.rescale = nn.ModuleList([
                nn.Sequential(nn.ConvTranspose2d(dim, dim // 2, 2, stride=2), nn.GroupNorm(32, dim // 2), nn.GELU(),
                              nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2)),  # stride 4
                nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 8
                nn.Identity(),  # stride 16
                nn.MaxPool2d(2),  # stride 32
            ])
            widths = (dim // 4, dim // 2, dim, dim)
        elif patch_size == 8:
            self.rescale = nn.ModuleList([
                nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),  # stride 4
                nn.Identity(),  # stride 8
                nn.MaxPool2d(2),  # stride 16
                nn.MaxPool2d(4),  # stride 32
            ])
            widths = (dim // 2, dim, dim, dim)
        else:
            raise ValueError(f'SimpleFeaturePyramid supports ViT strides 16 and 8, got {patch_size}')
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, channels, 1, bias=False), nn.GroupNorm(32, channels),
                          nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(32, channels))
            for c in widths)

    def forward(self, x):
        levels = [project(rescale(x)) for rescale, project in zip(self.rescale, self.project)]
        levels.append(F.max_pool2d(levels[-1], 1, stride=2))  # stride 64, as torchvision's FPN LastLevelMaxPool
        return OrderedDict(zip(['0', '1', '2', '3', 'pool'], levels))


class ConvStem(nn.Module):
    """A small conv stem on the raw input image: real stride-4 detail for the (otherwise upsampled) P2 level.

    The single-scale ViT only sees 16 px patches, so the neck's stride-4 level is upsampled and has no genuine
    high-frequency content. The stem reads the 512 px input directly and produces stride-4 features that are
    added to P2, which is what tiny SAR objects need. `channels` sets its width (the second conv has twice as
    many); `extra_convs` adds that many 3x3 convs at stride 4 (on `out_channels`) for a deeper stride-4 path.
    """

    def __init__(self, in_chans=3, channels=64, out_channels=256, extra_convs=0):
        super().__init__()
        layers = [
            nn.Conv2d(in_chans, channels, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, channels * 2), nn.GELU(),
            nn.Conv2d(channels * 2, out_channels, 1, bias=False), nn.GroupNorm(32, out_channels)]
        for _ in range(extra_convs):  # appended, so checkpoints without extra convs keep their layer indices
            layers += [nn.GELU(), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                       nn.GroupNorm(32, out_channels)]
        self.stem = nn.Sequential(*layers)

    def forward(self, x):
        return self.stem(x)


class InputAdapter(nn.Module):
    """ImageNet normalization followed by a learnable 1x1 conv, initialized as a no-op.

    DINOv3 was pretrained on ImageNet-normalized RGB; the conv lets the first layer remap the SAR pseudo-RGB
    channels into what the frozen patch embedding expects (the "train on the filtered input" idea).
    """

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    def __init__(self, in_chans=3):
        super().__init__()
        self.register_buffer('mean', torch.tensor(self.imagenet_mean[:in_chans]).view(1, in_chans, 1, 1))
        self.register_buffer('std', torch.tensor(self.imagenet_std[:in_chans]).view(1, in_chans, 1, 1))
        self.proj = nn.Conv2d(in_chans, in_chans, 1)
        with torch.no_grad():  # identity: starts as a no-op
            self.proj.weight.copy_(torch.eye(in_chans).view(in_chans, in_chans, 1, 1))
            self.proj.bias.zero_()

    def forward(self, x):
        return self.proj((x - self.mean) / self.std)


class WSTStem(nn.Module):
    """Wavelet scattering (J=2, L=8) -> 81 channels at stride 4, projected and added to the P2 level.

    Computed on the augmented pixels, it gives the tiny-object level noise-robust, multi-scale detail.
    """

    def __init__(self, out_channels=256, J=2, L=8, image_size=512):
        super().__init__()
        # Import the 2D frontend directly: kymatio 0.3's top-level import pulls in scattering3d, which breaks
        # on scipy >= 1.15 (removed sph_harm).
        from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D
        self.scattering = ScatteringTorch2D(J=J, L=L, shape=(image_size, image_size))
        n_coeffs = 1 + J * L + J * (J - 1) // 2 * L * L  # 81 for J=2, L=8
        self.head = nn.Sequential(
            nn.Conv2d(n_coeffs, out_channels, 1, bias=False), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.GroupNorm(32, out_channels))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):  # FFT scattering in fp32
            scattering = self.scattering(x[:, :1].float().contiguous())
        return self.head(scattering[:, 0].to(dtype=self.head[0].weight.dtype))


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


class CenterHeatmapHead(nn.Module):
    """Training-only auxiliary supervision of the stride-4 level (CenterNet, Zhou et al., 2019).

    Tiny objects get few positive anchors, so P2 and the stem receive little direct small-object signal. This head
    predicts one heatmap per class on P2 and is trained with CenterNet's penalty-reduced focal loss against a
    Gaussian peak at every GT centre (sigma = max(`min_sigma`, sqrt(w h) / (6 stride)) in P2 cells), so every object
    supervises P2 at its own location. It is never used at inference; it adds ~0.15 M trainable parameters.
    """

    def __init__(self, channels=256, hidden=64, num_classes=9, stride=4, min_sigma=0.8):
        super().__init__()
        self.stride, self.min_sigma = stride, min_sigma
        self.head = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(hidden, num_classes, 1))
        nn.init.constant_(self.head[-1].bias, -2.19)  # CenterNet's prior: sigmoid = 0.1 everywhere at init

    def targets(self, targets, num_classes, height, width, device):
        """[B, num_classes, height, width] Gaussian heatmaps, exactly 1 at each GT's centre cell."""
        ys = torch.arange(height, device=device, dtype=torch.float32).view(1, -1, 1)
        xs = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, -1)
        heatmaps = torch.zeros(len(targets), num_classes, height, width, device=device)
        for heatmap, target in zip(heatmaps, targets):
            boxes, labels = target['boxes'].float() / self.stride, target['labels']
            if not len(boxes):
                continue
            centers = ((boxes[:, :2] + boxes[:, 2:]) / 2).floor()  # the cell holding each centre
            centers[:, 0].clamp_(0, width - 1), centers[:, 1].clamp_(0, height - 1)
            sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
            sigma = (sizes.prod(1).sqrt() / 6).clamp(min=self.min_sigma).view(-1, 1, 1)
            peaks = torch.exp(-((xs - centers[:, :1, None]) ** 2 + (ys - centers[:, 1:, None]) ** 2)
                              / (2 * sigma ** 2))  # [K, H, W], 1 at each centre cell
            for label in labels.unique():
                heatmap[label - 1] = peaks[labels == label].amax(0)
        return heatmaps

    def loss(self, feature, targets):
        logits = self.head(feature)
        heatmaps = self.targets(targets, logits.shape[1], *logits.shape[-2:], logits.device)
        predicted = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
        positive = heatmaps.eq(1).float()
        positive_loss = (predicted.log() * (1 - predicted) ** 2 * positive).sum()
        negative_loss = ((1 - predicted).log() * predicted ** 2 * (1 - heatmaps) ** 4 * (1 - positive)).sum()
        return -(positive_loss + negative_loss) / positive.sum().clamp(min=1)


class MultiTaskModel(nn.Module):
    """backbone tokens -> classification logits (mean token -> LayerNorm -> Linear) and Faster R-CNN detections."""

    def __init__(self, backbone, num_classes=9, detail_stem=False, fusion_layers=None, giou_weight=0.0,
                 roi_sampling_ratio=2, roi_output_size=7, input_adapter=False, wst_stem=False, wst_params=None,
                 head='standard', head_params=None, detail_stem_params=None, rpn_fg_iou_thresh=0.7,
                 rpn_bg_iou_thresh=0.3, score_thresh=0.05, nms_thresh=0.5, detections_per_img=100,
                 vit_adapter=None, box_similarity='iou', nwd_constant=16.0, rescore='off', rescore_power=1.0,
                 iou_aware=False, iou_alpha=0.5, heatmap_aux=False, cls_detach=False):
        super().__init__()
        if rescore not in RESCORE_MODES:
            raise ValueError(f"unknown rescore mode '{rescore}'; available: {', '.join(RESCORE_MODES)}")
        if iou_aware and head != 'standard':
            raise ValueError('iou_aware is implemented for the standard RoI head only')
        self.backbone = backbone
        self.patch_size = getattr(backbone, 'patch_size', 16)
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        # cls_detach: the classifier reads the pooled tokens without back-propagating into the backbone, so DoRA,
        # Convpass and the input adapter train for detection only (mAP is worth ~3.7x macro-F1 per point in delta-m).
        self.cls_detach = cls_detach
        self.neck = SimpleFeaturePyramid(backbone.embed_dim, patch_size=self.patch_size)
        self.detail_stem = ConvStem(backbone.in_chans, **(detail_stem_params or {})) if detail_stem else None
        self.input_adapter = InputAdapter(backbone.in_chans) if input_adapter else None
        self.wst_stem = WSTStem(**(wst_params or {})) if wst_stem else None
        if input_adapter and hasattr(backbone, 'unfreeze_patch_embed'):  # adapt_backbone froze it; re-enable it
            backbone.unfreeze_patch_embed()
        self.heatmap_head = CenterHeatmapHead(num_classes=num_classes) if heatmap_aux else None
        self.vit_adapter = None
        if vit_adapter:  # wraps the interaction blocks, after DoRA / Convpass (adapt_backbone) wrapped them
            self.vit_adapter = ViTAdapter(backbone.embed_dim, getattr(backbone, 'num_prefix_tokens', 0),
                                          self.patch_size, in_chans=backbone.in_chans, **vit_adapter)
            self.vit_adapter.attach(backbone)
        # Multi-layer fusion: normalize negative indices, register hooks in execution order.
        self.fusion_layers = self.fusion = None
        if fusion_layers:
            blocks = len(backbone.blocks)
            self.fusion_layers = sorted(i % blocks for i in fusion_layers)
            self.fusion = LayerFusion(len(self.fusion_layers))
        # Small objects (median ~16 px at 512 px, 99 % below ~240 px): two anchor scales per octave, from 8 px at
        # stride 4 up to 181 px at stride 64, times 3 aspect ratios = 6 anchors per location. An anchor is an RPN
        # positive at IoU >= rpn_fg_iou_thresh (a few px of offset push a 6-15 px box below 0.7; every GT still keeps
        # its best anchor) and a negative below rpn_bg_iou_thresh.
        anchors = AnchorGenerator(((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), ((0.5, 1.0, 2.0),) * 5)
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(256, 6), fg_iou_thresh=rpn_fg_iou_thresh, bg_iou_thresh=rpn_bg_iou_thresh,
            batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=2000),
            post_nms_top_n=dict(training=2000, testing=2000), nms_thresh=0.7)
        roi_pool = MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=roi_output_size,
                                      sampling_ratio=roi_sampling_ratio)
        self.roi_heads = self._build_roi_heads(roi_pool, roi_output_size, num_classes, head, head_params or {},
                                               giou_weight, score_thresh, nms_thresh, detections_per_img,
                                               iou_aware, iou_alpha)
        self.roi_heads.rescore, self.roi_heads.rescore_power = rescore, rescore_power
        # Anchor (RPN) and proposal (RoI head) labels: IoU, NWD or their max against the same thresholds.
        self.rpn.box_similarity = self.roi_heads.box_similarity = make_similarity(box_similarity, nwd_constant)

    @staticmethod
    def _build_roi_heads(roi_pool, resolution, num_classes, head, params, giou_weight, score_thresh, nms_thresh,
                         detections_per_img, iou_aware=False, iou_alpha=0.5):
        """`standard`: one Faster R-CNN RoI head. `cascade`: Cascade R-CNN with `head_params` num_stages (3),
        share_head (true), fg_iou_thresholds (0.5/0.6/...), bg_iou_threshold (0.5) and stage_loss_weights (1.0).

        Both take `head_params` box_head: `mlp` (torchvision's two fc layers of width hidden_dim, default 1024) or
        `conv` (ConvFCBoxHead: num_convs 3x3 convs, a 1x1 down to `reduce` channels, one fc of width hidden_dim).
        score_thresh / nms_thresh / detections_per_img are the inference post-processing (no effect on training).
        `iou_aware` adds the standard head's IoU branch; `iou_alpha` is its inference ranking exponent.
        """
        hidden = params.get('hidden_dim', 1024)
        post = dict(score_thresh=score_thresh, nms_thresh=nms_thresh, detections_per_img=detections_per_img)

        def box_head():
            if params.get('box_head', 'mlp') == 'mlp':
                return TwoMLPHead(256 * resolution ** 2, hidden)
            if params['box_head'] == 'conv':
                keys = ('num_convs', 'conv_dim', 'reduce')
                return ConvFCBoxHead(256, resolution, fc_dim=hidden, **{k: params[k] for k in keys if k in params})
            raise ValueError(f"unknown box_head '{params['box_head']}'; available: mlp, conv")

        if head == 'standard':
            return DetRoIHeads(
                roi_pool, box_head(), FastRCNNPredictor(hidden, num_classes + 1),
                fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
                bbox_reg_weights=None, giou_weight=giou_weight, iou_dim=hidden if iou_aware else None,
                iou_alpha=iou_alpha, **post)
        if head == 'cascade':
            stages = params.get('num_stages', 3)
            # A per-stage head at hidden 1024 costs ~14 M each; hidden 256 per stage costs less than one 1024 head.
            box_heads = [box_head() for _ in range(1 if params.get('share_head', True) else stages)]
            predictors = [FastRCNNPredictor(hidden, num_classes + 1) for _ in range(stages)]
            thresholds = tuple(params.get('fg_iou_thresholds', ())) \
                or tuple(round(0.5 + 0.1 * i, 2) for i in range(stages))
            return CascadeRoIHeads(
                roi_pool, box_heads, predictors, fg_iou_thresholds=thresholds,
                bg_iou_threshold=params.get('bg_iou_threshold', 0.5), batch_size_per_image=512,
                positive_fraction=0.25, bbox_reg_weights=None, stage_loss_weights=params.get('stage_loss_weights'),
                giou_weight=giou_weight, **post)
        raise ValueError(f"unknown head '{head}'; available: standard, cascade")

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
        backbone_images = self.input_adapter(images) if self.input_adapter is not None else images
        if self.vit_adapter is not None:  # the spatial prior runs first: its injectors act inside the backbone
            self.vit_adapter.prepare(images)
        tokens = self._backbone_tokens(backbone_images)
        pooled = tokens.mean(dim=1)
        logits = self.cls_head(pooled.detach() if self.cls_detach else pooled)
        h, w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size  # one token per patch
        features = self.neck(tokens.transpose(1, 2).unflatten(2, (h, w)))  # tokens as a [B, 768, h, w] map
        if self.vit_adapter is not None:  # the SPM features (strides 4-32) added to P2-P5
            for name, extra in zip(('0', '1', '2', '3'), self.vit_adapter.outputs()):
                features[name] = features[name] + extra.to(features[name].dtype)
        if self.detail_stem is not None:  # add genuine stride-4 detail (from the pixels) to the upsampled P2
            features['0'] = features['0'] + self.detail_stem(images.to(features['0'].dtype))
        if self.wst_stem is not None:  # scattering detail (81 coeffs at stride 4) added to P2
            features['0'] = features['0'] + self.wst_stem(images).to(features['0'].dtype)
        # Precision rule: the box coder casts anchors to the dtype of the regression output, so in bf16 every
        # coordinate near 512 would snap to a 2 px grid. RPN and RoIHeads therefore run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            if self.training and self.heatmap_head is not None:  # training-only P2 supervision
                rpn_losses['loss_heatmap'] = self.heatmap_head.loss(features['0'], targets)
            # At inference the image classifier's probabilities re-score the boxes (roi_heads.rescore, off by default).
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets,
                                                    class_prior=None if self.training else logits.float().softmax(-1))
        return logits, detections, {**rpn_losses, **roi_losses}


def build_model(args):
    """DINOv3 + DoRA (+ the per-block capacity options) + heads, on the CPU. Returns (model, ids of pretrained
    backbone parameters), so the optimizer can give unfrozen pretrained weights their own learning rate."""
    backbone = DINOv3(pretrained=args.pretrained, **args.backbone_kwargs)
    pretrained_ids = {id(p) for p in backbone.parameters()}
    conv = args.conv_adapters or {}
    adapt_backbone(backbone, targets=args.peft_targets, rank=args.peft_kwargs.get('rank', 32),
                   alpha=args.peft_kwargs.get('alpha', 32), rank_overrides=args.rank_overrides,
                   full_blocks=args.full_blocks, conv_layers=conv.get('layers', ()), conv_dim=conv.get('dim', 128),
                   conv_kernel=conv.get('kernel', 'dense'), conv_placement=conv.get('placement', 'sequential'))
    model = MultiTaskModel(backbone, detail_stem=args.detail_stem, fusion_layers=args.fusion_layers,
                           giou_weight=args.giou_weight, roi_sampling_ratio=args.roi_sampling_ratio,
                           roi_output_size=args.roi_output_size, input_adapter=args.input_adapter,
                           wst_stem=args.wst_stem, wst_params=args.wst_params, head=args.head,
                           head_params=args.head_params, detail_stem_params=args.detail_stem_params,
                           rpn_fg_iou_thresh=args.rpn_fg_iou_thresh, rpn_bg_iou_thresh=args.rpn_bg_iou_thresh,
                           score_thresh=args.score_thresh, nms_thresh=args.nms_thresh,
                           detections_per_img=args.detections_per_img, vit_adapter=args.vit_adapter,
                           box_similarity=args.box_similarity, nwd_constant=args.nwd_constant,
                           rescore=args.rescore, rescore_power=args.rescore_power, iou_aware=args.iou_aware,
                           iou_alpha=args.iou_alpha, heatmap_aux=args.heatmap_aux, cls_detach=args.cls_detach)
    return model, pretrained_ids


def parameter_counts(model, pretrained_ids):
    """Total and trainable parameters, plus the trainable breakdown: heads / new backbone / unfrozen pretrained."""
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    return {
        'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'heads': sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone.')),
        'backbone_new': sum(p.numel() for p in backbone if id(p) not in pretrained_ids),
        'backbone_unfrozen': sum(p.numel() for p in backbone if id(p) in pretrained_ids),
    }
