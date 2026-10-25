"""RoI heads with an optional GIoU box-regression term: torchvision's Faster R-CNN head (`DetRoIHeads`) and
Cascade R-CNN (`CascadeRoIHeads`), plus a conv box head (`ConvFCBoxHead`) usable by either. Both RoI heads return
the same `(per-image detections, loss dict)` interface as torchvision, so the rest of the pipeline is unchanged.

Both heads also take the image classifier's probabilities at inference (`class_prior`) and re-score their boxes
with them (`rescore`, see `apply_class_prior`), and match proposals to GT with a configurable `box_similarity`
(IoU, NWD or the larger of the two, see `box_similarity`). The standard head can also predict each box's IoU with its
ground truth and rank detections by class score x predicted IoU (`iou_aware`, see `apply_box_quality`).
"""
import functools

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler, Matcher
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import box_iou, clip_boxes_to_image

from .losses import fastrcnn_loss as roi_loss

RESCORE_MODES = ('off', 'soft', 'hard', 'merge')
BOX_SIMILARITIES = ('iou', 'nwd', 'max')


def normalized_wasserstein(boxes1, boxes2, constant=16.0):
    """NWD (Wang et al., 2021, tiny-object detection): exp(-W2 / C) between XYXY boxes, [N, M] in (0, 1].

    Each box is the 2-D Gaussian N((cx, cy), diag(w^2 / 4, h^2 / 4)), whose 2-Wasserstein distance is the Euclidean
    distance between (cx, cy, w / 2, h / 2). Unlike IoU, it decays smoothly with a few px of offset, so a 6-15 px box
    shifted by 2 px still matches its anchor. `constant` (C) is about the typical box size in px (median 15.9 here).
    """
    def gaussian(boxes):
        return torch.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
                            (boxes[:, 2] - boxes[:, 0]) / 2, (boxes[:, 3] - boxes[:, 1]) / 2], dim=1)
    return torch.exp(-torch.cdist(gaussian(boxes1.float()), gaussian(boxes2.float())) / constant)


def _iou_or_nwd(boxes1, boxes2, constant):
    return torch.maximum(box_iou(boxes1, boxes2), normalized_wasserstein(boxes1, boxes2, constant))


def box_similarity(name='iou', nwd_constant=16.0):
    """The [GT, box] similarity that labels anchors (RPN) and proposals (RoI head) against the IoU thresholds:
    `iou` (torchvision's), `nwd` (`normalized_wasserstein`) or `max` (the larger of the two: NWD's leniency for tiny
    boxes, where it exceeds IoU, while large boxes keep their IoU labels)."""
    if name == 'iou':
        return box_iou
    if name == 'nwd':
        return functools.partial(normalized_wasserstein, constant=nwd_constant)
    if name == 'max':
        return functools.partial(_iou_or_nwd, constant=nwd_constant)
    raise ValueError(f"unknown box_similarity '{name}'; available: {', '.join(BOX_SIMILARITIES)}")


def apply_class_prior(class_logits, prior, mode='soft', power=1.0):
    """Re-score the RoI head's class scores with the image classifier's probabilities (inference only).

    Every image has one class and all its boxes share it, so a box of another class is a false positive for that
    class. `class_logits` [N, 1 + C] (background first), `prior` [N, C] (the image's probabilities, one row per
    proposal). `mode`:

        soft    p(c | box) * p(c | image) ** power  (power 0 changes nothing; large powers approach hard)
        hard    only the image's predicted class keeps its score; every other class is zeroed
        merge   the box's whole foreground mass, 1 - p(background | box), goes to the image's predicted class

    Returns logits whose softmax is the re-scored distribution (the background takes the remainder), so torchvision's
    post-processing (score threshold, per-class NMS, top-k) runs on the new scores unchanged.
    """
    scores = class_logits.float().softmax(-1)
    foreground = scores[:, 1:]
    if mode == 'soft':
        foreground = foreground * prior.float().pow(power)
    elif mode in ('hard', 'merge'):
        predicted = F.one_hot(prior.argmax(1), prior.shape[1]).to(foreground.dtype)
        foreground = predicted * (foreground.sum(1, keepdim=True) if mode == 'merge' else foreground)
    elif mode != 'off':
        raise ValueError(f"unknown rescore mode '{mode}'; available: {', '.join(RESCORE_MODES)}")
    return _as_logits(foreground)


def apply_box_quality(class_logits, quality, alpha=0.5):
    """Rank boxes by how well they are localized: p(c | box) * IoU_c ** alpha (inference only).

    `class_logits` [N, 1 + C], `quality` [N, C] the predicted IoU of each class's decoded box with its GT (in [0, 1]).
    The class score alone says nothing about localization, so a tight box can rank below a loose one; AP75 and above
    pay for that. alpha 0 changes nothing (the same model ranked by class score alone). Returns logits, as
    `apply_class_prior` does, so the two compose and torchvision's post-processing is unchanged.
    """
    foreground = class_logits.float().softmax(-1)[:, 1:] * quality.float().clamp(min=0).pow(alpha)
    return _as_logits(foreground)


def _as_logits(foreground):
    """Logits whose softmax is `foreground` [N, C] with the background taking the remainder (re-scored foreground
    mass never exceeds the original, so the remainder is >= 0)."""
    background = (1 - foreground.sum(1, keepdim=True)).clamp(min=0)
    return torch.cat([background, foreground], dim=1).clamp(min=1e-12).log()


def paired_iou(boxes1, boxes2):
    """IoU of each row of `boxes1` [N, 4] with the same row of `boxes2` (XYXY): [N]."""
    top_left = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    inter = (bottom_right - top_left).clamp(min=0).prod(1)
    area = lambda b: (b[:, 2:] - b[:, :2]).clamp(min=0).prod(1)
    return inter / (area(boxes1) + area(boxes2) - inter).clamp(min=1e-6)


def _rescored(heads, class_logits, proposals, class_prior):
    """`class_logits` re-scored with `class_prior` [B, C] when the heads' `rescore` is on (else unchanged)."""
    if class_prior is None or heads.rescore == 'off':
        return class_logits
    rows = torch.cat([p.expand(len(b), -1) for p, b in zip(class_prior, proposals)])
    return apply_class_prior(class_logits, rows, heads.rescore, heads.rescore_power)


class ConvFCBoxHead(nn.Module):
    """RoI box head of `num_convs` 3x3 convs (GroupNorm, ReLU) on the pooled 256 x 7 x 7 features, a 1x1 conv down to
    `reduce` channels, then one fully connected layer of width `fc_dim` (the "4conv1fc" head, made cheaper).

    The 3x3 convs keep the RoI's spatial layout for localization, and the 1x1 reduction makes the fully connected
    layer `reduce`/256 of torchvision's first fc: 5.6 M parameters instead of 13.9 M at the defaults.
    """

    def __init__(self, in_channels=256, resolution=7, num_convs=4, conv_dim=256, reduce=64, fc_dim=1024):
        super().__init__()
        layers, channels = [], in_channels
        for _ in range(num_convs):
            layers += [nn.Conv2d(channels, conv_dim, 3, padding=1, bias=False), nn.GroupNorm(32, conv_dim),
                       nn.ReLU(inplace=True)]
            channels = conv_dim
        layers += [nn.Conv2d(channels, reduce, 1), nn.ReLU(inplace=True)]
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(reduce * resolution * resolution, fc_dim)

    def forward(self, x):
        return torch.relu(self.fc(self.convs(x).flatten(1)))


class DetRoIHeads(RoIHeads):
    """torchvision's RoI head with an additional GIoU box-regression term (`giou_weight`, 0 disables it), image-class
    re-scoring at inference (`rescore`, `rescore_power`) and proposal matching by `self.box_similarity`.

    `iou_dim` (the box head's output width; None disables it) adds an IoU branch: one linear layer predicting, per
    class, the IoU of that class's decoded box with its GT (IoU-Net, Jiang et al., 2018). It trains on the sampled
    positives (`loss_iou`: binary cross-entropy against the detached IoU of the box as currently regressed), and at
    inference detections are ranked by class score x predicted IoU ** `iou_alpha` (`apply_box_quality`).
    """

    def __init__(self, *args, giou_weight=0.0, rescore='off', rescore_power=1.0, iou_dim=None, iou_alpha=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.giou_weight = giou_weight
        self.rescore, self.rescore_power = rescore, rescore_power
        num_classes = self.box_predictor.cls_score.out_features - 1
        self.iou_predictor = nn.Linear(iou_dim, num_classes) if iou_dim else None
        self.iou_alpha = iou_alpha

    def iou_loss(self, iou_logits, box_regression, proposals, labels, matched_gt):
        """BCE of the predicted IoU of each positive's GT-class box against that box's actual IoU with its GT."""
        labels = torch.cat(labels)
        positive = torch.where(labels > 0)[0]
        if not len(positive):
            return iou_logits.sum() * 0
        decoded = self.box_coder.decode(box_regression.detach(), proposals)  # [N, 1 + C, 4]
        label = labels[positive]
        predicted = decoded[positive, label]
        target = paired_iou(predicted, torch.cat(matched_gt)[positive].to(predicted.dtype))
        return F.binary_cross_entropy_with_logits(iou_logits[positive, label - 1], target)

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        """torchvision's, with `self.box_similarity` in place of its hard-coded box_iou."""
        matched_idxs, labels = [], []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # background image
                matched = torch.zeros((len(proposals_in_image),), dtype=torch.int64, device=proposals_in_image.device)
                labels_in_image = torch.zeros_like(matched)
            else:
                raw = self.proposal_matcher(self.box_similarity(gt_boxes_in_image, proposals_in_image))
                matched = raw.clamp(min=0)
                labels_in_image = gt_labels_in_image[matched].to(torch.int64)
                labels_in_image[raw == self.proposal_matcher.BELOW_LOW_THRESHOLD] = 0  # background
                labels_in_image[raw == self.proposal_matcher.BETWEEN_THRESHOLDS] = -1  # ignored by the sampler
            matched_idxs.append(matched)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def forward(self, features, proposals, image_shapes, targets=None, class_prior=None):
        matched_gt = None
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            if self.giou_weight > 0 or self.iou_predictor is not None:  # an image may have no boxes left after
                # augmentation: no matched GT then
                matched_gt = [t['boxes'][mi.clamp(min=0)].to(proposals[0].dtype) if len(t['boxes'])
                              else proposals[0].new_zeros((0, 4)) for t, mi in zip(targets, matched_idxs)]
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        iou_logits = self.iou_predictor(box_features) if self.iou_predictor is not None else None
        detections, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = roi_loss(class_logits, box_regression, labels, regression_targets,
                                                     proposals, matched_gt, self.box_coder,
                                                     giou_weight=self.giou_weight)
            losses = {'loss_classifier': loss_classifier, 'loss_box_reg': loss_box_reg}
            if iou_logits is not None:
                losses['loss_iou'] = self.iou_loss(iou_logits, box_regression, proposals, labels, matched_gt)
        else:
            if iou_logits is not None:  # localization quality first, then the image-class prior on top
                class_logits = apply_box_quality(class_logits, iou_logits.float().sigmoid(), self.iou_alpha)
            all_boxes, all_scores, all_labels = self.postprocess_detections(
                _rescored(self, class_logits, proposals, class_prior), box_regression, proposals, image_shapes)
            detections = [{'boxes': b, 'labels': l, 'scores': s}
                          for b, s, l in zip(all_boxes, all_scores, all_labels)]
        return detections, losses


class CascadeRoIHeads(RoIHeads):
    """Cascade R-CNN (Cai & Vasconcelos, 2018): S RoI stages with increasing IoU thresholds, each decoding
    boxes that are re-pooled and refined by the next stage. Reuses torchvision's box coder and post-processing;
    the per-stage losses are summed into `loss_classifier` / `loss_box_reg` so the `loss:` weights still apply.
    """

    def __init__(self, box_roi_pool, box_heads, box_predictors, fg_iou_thresholds=(0.5, 0.6, 0.7),
                 bg_iou_threshold=0.5, batch_size_per_image=512, positive_fraction=0.25, bbox_reg_weights=None,
                 score_thresh=0.05, nms_thresh=0.5, detections_per_img=100, stage_loss_weights=None,
                 giou_weight=0.0, rescore='off', rescore_power=1.0):
        # Matcher requires low <= high, so the background threshold must not exceed the smallest fg threshold.
        bg_iou_threshold = min(bg_iou_threshold, min(fg_iou_thresholds))
        # The base RoIHeads supplies the box coder, the RoI pool and post-processing shared by all stages.
        super().__init__(box_roi_pool, box_heads[0], box_predictors[0], fg_iou_thresholds[0], bg_iou_threshold,
                         batch_size_per_image, positive_fraction, bbox_reg_weights, score_thresh, nms_thresh,
                         detections_per_img)
        self.box_heads = nn.ModuleList(box_heads)
        self.box_predictors = nn.ModuleList(box_predictors)
        # Matcher / BalancedPositiveNegativeSampler are plain (parameter-free) helpers, not nn.Modules.
        self.cascade_matchers = [Matcher(t, bg_iou_threshold) for t in fg_iou_thresholds]
        self.cascade_samplers = [
            BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction) for _ in fg_iou_thresholds]
        self.stage_loss_weights = list(stage_loss_weights) if stage_loss_weights else [1.0] * len(box_predictors)
        self.giou_weight = giou_weight
        self.rescore, self.rescore_power = rescore, rescore_power

    def _refined_boxes(self, box_regression, class_logits, boxes, image_shapes):
        """Decode one box per proposal (highest-scoring foreground class) for the next cascade stage.

        `BoxCoder.decode` returns a single `[N, num_classes, 4]` tensor, so the per-proposal, per-image
        reduction has to be done explicitly (zipping it with `image_shapes` would treat rows as images).
        """
        decoded = self.box_coder.decode(box_regression, boxes)  # [N, num_classes, 4]
        class_idx = class_logits[:, 1:].argmax(dim=-1) + 1  # best foreground class per proposal
        rows = torch.arange(decoded.shape[0], device=decoded.device)
        decoded = decoded[rows, class_idx]  # [N, 4]
        counts = [len(b) for b in boxes]
        return [clip_boxes_to_image(b, s) for b, s in zip(decoded.split(counts, dim=0), image_shapes)]

    def _sample(self, proposals, targets, matcher, sampler):
        """Match proposals to targets and sample positives/negatives for one cascade stage (as RoIHeads does)."""
        gt_boxes = [t['boxes'] for t in targets]
        gt_labels = [t['labels'] for t in targets]
        proposals_with_gt = [torch.cat((p, g.to(p.dtype))) for p, g in zip(proposals, gt_boxes)]
        # Matcher expects an M(gt) x N(predicted) IoU matrix; it returns -1 (below) / -2 (between) / >=0.
        matched = [matcher(self.box_similarity(g.to(p.dtype), p)) for p, g in zip(proposals_with_gt, gt_boxes)]
        clamped = [m.clamp(min=0) for m in matched]
        labels = []
        for m, clamped_idxs, gt_label in zip(matched, clamped, gt_labels):
            label = gt_label[clamped_idxs].to(torch.int64)
            label[m == -1] = 0  # background (below the low threshold)
            label[m == -2] = -1  # between thresholds: ignored by the sampler
            labels.append(label)
        pos_idx, neg_idx = sampler(labels)
        inds = [torch.where(p | n)[0] for p, n in zip(pos_idx, neg_idx)]
        sampled_boxes = [p[i] for p, i in zip(proposals_with_gt, inds)]
        sampled_labels = [l[i] for l, i in zip(labels, inds)]
        sampled_clamped = [c[i] for c, i in zip(clamped, inds)]
        # BoxCoder.encode takes a list of per-image tensors (negative targets are ignored by fastrcnn_loss).
        matched_gt = [g[cm] if len(g) else g.new_zeros((0, 4)) for g, cm in zip(gt_boxes, sampled_clamped)]
        regression_targets = self.box_coder.encode(matched_gt, sampled_boxes)
        return sampled_boxes, sampled_labels, regression_targets, matched_gt

    def forward(self, features, proposals, image_shapes, targets=None, class_prior=None):
        boxes, labels, regression_targets, matched_gt = proposals, None, None, None
        if self.training:
            boxes, labels, regression_targets, matched_gt = self._sample(
                proposals, targets, self.cascade_matchers[0], self.cascade_samplers[0])
        losses = {}
        detections = []
        for stage, predictor in enumerate(self.box_predictors):
            head = self.box_heads[min(stage, len(self.box_heads) - 1)]  # shared head unless per-stage provided
            if stage > 0 and self.training:
                boxes, labels, regression_targets, matched_gt = self._sample(
                    boxes, targets, self.cascade_matchers[stage], self.cascade_samplers[stage])
            # At inference the refined boxes from the previous stage carry over as this stage's proposals.
            pooled = self.box_roi_pool(features, boxes, image_shapes)
            class_logits, box_regression = predictor(head(pooled))
            if self.training:
                loss_classifier, loss_box_reg = roi_loss(class_logits, box_regression, labels,
                                                         regression_targets, boxes, matched_gt, self.box_coder,
                                                         giou_weight=self.giou_weight)
                weight = self.stage_loss_weights[stage]
                losses['loss_classifier'] = losses.get('loss_classifier', 0.) + weight * loss_classifier
                losses['loss_box_reg'] = losses.get('loss_box_reg', 0.) + weight * loss_box_reg
                boxes = self._refined_boxes(box_regression.detach(), class_logits, boxes, image_shapes)
            elif stage < len(self.box_predictors) - 1:
                # Keep every refined proposal between stages (no score threshold / NMS): the later stages need
                # high-recall candidates for small objects. Only the final stage thresholds and suppresses.
                boxes = self._refined_boxes(box_regression, class_logits, boxes, image_shapes)
            else:
                all_boxes, all_scores, all_labels = self.postprocess_detections(
                    _rescored(self, class_logits, boxes, class_prior), box_regression, boxes, image_shapes)
                detections = [{'boxes': b, 'labels': l, 'scores': s}
                              for b, s, l in zip(all_boxes, all_scores, all_labels)]
        return detections, losses
