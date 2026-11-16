"""The multi-task model: one DINOv3 backbone feeds an image classifier and a Faster R-CNN detector.

    pseudo-RGB [B, 3, 512, 512] -> input adapter -> DINOv3 ViT-B/16 (frozen) + DoRA + Convpass -> tokens [B, 1024, 768]
      classification: mean token -> LayerNorm -> Linear(768, 9)
      detection:      tokens as a 32 x 32 map -> simple feature pyramid P2-P6 (+ P2 conv stem on the pixels)
                      -> RPN -> fused RoIAlign (max over P2-P5) -> 2-fc box head -> boxes, GIoU term in the loss
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import roi_align

from .adapters import adapt_backbone
from .backbones import DINOv3
from .heads import DetRoIHeads

# Competition limits (Codabench terms): every parameter counts towards the total, frozen ones included.
MAX_TOTAL_PARAMETERS = 130_000_000
MAX_TRAINABLE_PARAMETERS = 40_000_000
NUM_CLASSES = 9


class SimpleFeaturePyramid(nn.Module):
    """ViTDet neck (Li et al., 2022): the stride-16 ViT map -> 256-channel maps at strides 4, 8, 16, 32, 64."""

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
    """A small conv stem on the input pixels whose stride-4 output is added to P2.

    The ViT sees only 16 px patches, so the neck's P2 is upsampled and has no real stride-4 detail; half of the boxes
    are smaller than one patch. Two stride-2 3x3 convs (`channels`, then twice as many), a 1x1 to `out_channels`,
    then `extra_convs` more 3x3 convs at stride 4.
    """

    def __init__(self, in_chans=3, channels=128, out_channels=256, extra_convs=1):
        super().__init__()
        layers = [
            nn.Conv2d(in_chans, channels, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, channels * 2), nn.GELU(),
            nn.Conv2d(channels * 2, out_channels, 1, bias=False), nn.GroupNorm(32, out_channels)]
        for _ in range(extra_convs):
            layers += [nn.GELU(), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                       nn.GroupNorm(32, out_channels)]
        self.stem = nn.Sequential(*layers)

    def forward(self, x):
        return self.stem(x)


class InputAdapter(nn.Module):
    """ImageNet normalization, then a learnable 1x1 conv initialized to the identity: lets the first layer remap the
    SAR pseudo-RGB channels into what the (unfrozen) DINOv3 patch embedding expects."""

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    def __init__(self, in_chans=3):
        super().__init__()
        self.register_buffer('mean', torch.tensor(self.imagenet_mean[:in_chans]).view(1, in_chans, 1, 1))
        self.register_buffer('std', torch.tensor(self.imagenet_std[:in_chans]).view(1, in_chans, 1, 1))
        self.proj = nn.Conv2d(in_chans, in_chans, 1)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(in_chans).view(in_chans, in_chans, 1, 1))
            self.proj.bias.zero_()

    def forward(self, x):
        return self.proj((x - self.mean) / self.std)


class FusedRoIAlign(nn.Module):
    """Pool every RoI from all of P2-P5 and fuse them by element-wise max (PANet's adaptive feature pooling).

    torchvision's mapping (canonical 224 px at P4) sends every box under 112 px, 94.9 % of this dataset's boxes, to
    P2 alone, so they never read P4, the ViT's own stride-16 map. Fusing gives each RoI P2's detail and P3-P5's
    semantics, with no parameters.
    """

    def __init__(self, featmap_names=('0', '1', '2', '3'), output_size=7, sampling_ratio=2):
        super().__init__()
        self.featmap_names, self.output_size, self.sampling_ratio = list(featmap_names), output_size, sampling_ratio

    def forward(self, features, boxes, image_shapes):
        rois = torch.cat([torch.cat([b.new_full((len(b), 1), i), b], dim=1) for i, b in enumerate(boxes)])
        pooled = [roi_align(features[name], rois, self.output_size, spatial_scale=features[name].shape[-1] /
                            image_shapes[0][1], sampling_ratio=self.sampling_ratio, aligned=False)
                  for name in self.featmap_names]
        return torch.stack(pooled).amax(0)


class MultiTaskModel(nn.Module):
    """Backbone tokens -> classification logits and Faster R-CNN detections (see the module docstring)."""

    def __init__(self, backbone, num_classes=NUM_CLASSES, stem_channels=128, stem_extra_convs=1, giou_weight=1.0,
                 score_thresh=1e-5, nms_thresh=0.85, detections_per_img=100):
        super().__init__()
        # The construction order fixes the random initialization: keep it.
        self.backbone = backbone
        self.patch_size = backbone.patch_size
        self.cls_head = nn.Sequential(nn.LayerNorm(backbone.embed_dim), nn.Linear(backbone.embed_dim, num_classes))
        self.neck = SimpleFeaturePyramid(backbone.embed_dim)
        self.detail_stem = ConvStem(backbone.in_chans, stem_channels, extra_convs=stem_extra_convs)
        self.input_adapter = InputAdapter(backbone.in_chans)
        backbone.unfreeze_patch_embed()
        # Small objects (median 15.9 px at 512 px): two anchor sizes per octave, 8 px at stride 4 up to 181 px at
        # stride 64, times 3 aspect ratios = 6 anchors per location.
        anchors = AnchorGenerator(((8, 11), (16, 23), (32, 45), (64, 91), (128, 181)), ((0.5, 1.0, 2.0),) * 5)
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(256, 6), fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256,
            positive_fraction=0.5, pre_nms_top_n=dict(training=2000, testing=2000),
            post_nms_top_n=dict(training=2000, testing=2000), nms_thresh=0.7)
        self.roi_heads = DetRoIHeads(
            FusedRoIAlign(), TwoMLPHead(256 * 7 ** 2, 1024), FastRCNNPredictor(1024, num_classes + 1),
            fg_iou_thresh=0.5, bg_iou_thresh=0.5, batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None, score_thresh=score_thresh, nms_thresh=nms_thresh,
            detections_per_img=detections_per_img, giou_weight=giou_weight)

    def forward(self, images, targets=None):
        """images [B, 3, H, W], targets: per-image {'boxes' XYXY, 'labels' 1..9} (training only) ->
        (logits [B, num_classes], per-image detections (eval mode), the detection losses (training mode))."""
        tokens = self.backbone(self.input_adapter(images))
        logits = self.cls_head(tokens.mean(dim=1))
        h, w = images.shape[-2] // self.patch_size, images.shape[-1] // self.patch_size
        features = self.neck(tokens.transpose(1, 2).unflatten(2, (h, w)))  # tokens as a [B, 768, h, w] map
        features['0'] = features['0'] + self.detail_stem(images.to(features['0'].dtype))
        # The box coder casts anchors to the dtype of the regression output, so in bf16 every coordinate near 512
        # would snap to a 2 px grid: the RPN and RoI head run in fp32.
        with torch.autocast(images.device.type, enabled=False):
            features = {name: f.float() for name, f in features.items()}
            image_list = ImageList(images, [tuple(images.shape[-2:])] * len(images))
            proposals, rpn_losses = self.rpn(image_list, features, targets)
            detections, roi_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)
        return logits, detections, {**rpn_losses, **roi_losses}


def build_model(config, pretrained=True):
    """The final model from a loaded config, on the CPU. `pretrained` loads the DINOv3 checkpoint (training); False
    builds the same architecture with random weights, to be replaced by a saved model.pt (inference). Returns
    (model, ids of the pretrained backbone parameters), so the optimizer can give the unfrozen ones their own rate."""
    adapters, model = config['adapters'], config['model']
    backbone = DINOv3(config['backbone']['model_name'], pretrained=pretrained)
    pretrained_ids = {id(p) for p in backbone.parameters()}
    adapt_backbone(backbone, targets=adapters['dora_targets'], rank=adapters['dora_rank'],
                   alpha=adapters['dora_alpha'], convpass_dim=adapters['convpass_dim'])
    return MultiTaskModel(backbone, stem_channels=model['stem_channels'], stem_extra_convs=model['stem_extra_convs'],
                          giou_weight=model['giou_weight'], score_thresh=model['score_thresh'],
                          nms_thresh=model['nms_thresh'],
                          detections_per_img=model['detections_per_img']), pretrained_ids


def parameter_counts(model, pretrained_ids):
    """Total and trainable parameters, plus the trainable breakdown: heads / new backbone / unfrozen pretrained."""
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    return {
        'total': sum(p.numel() for p in model.parameters()),
        'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'heads': sum(p.numel() for n, p in model.named_parameters()
                     if p.requires_grad and not n.startswith('backbone.')),
        'backbone_new': sum(p.numel() for p in backbone if id(p) not in pretrained_ids),
        'backbone_unfrozen': sum(p.numel() for p in backbone if id(p) in pretrained_ids),
    }
