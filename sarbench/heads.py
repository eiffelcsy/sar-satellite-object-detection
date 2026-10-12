"""RoI heads with an optional GIoU box-regression term: torchvision's Faster R-CNN head (`DetRoIHeads`) and
Cascade R-CNN (`CascadeRoIHeads`). Both return the same `(per-image detections, loss dict)` interface as
torchvision, so the rest of the pipeline is unchanged.
"""
import torch
from torch import nn
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler, Matcher
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import box_iou, clip_boxes_to_image

from .losses import fastrcnn_loss as roi_loss


class DetRoIHeads(RoIHeads):
    """torchvision's RoI head with an additional GIoU box-regression term (`giou_weight`, 0 disables it)."""

    def __init__(self, *args, giou_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.giou_weight = giou_weight

    def forward(self, features, proposals, image_shapes, targets=None):
        matched_gt = None
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            if self.giou_weight > 0:  # an image may have no boxes left after augmentation: no matched GT then
                matched_gt = [t['boxes'][mi.clamp(min=0)].to(proposals[0].dtype) if len(t['boxes'])
                              else proposals[0].new_zeros((0, 4)) for t, mi in zip(targets, matched_idxs)]
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        detections, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = roi_loss(class_logits, box_regression, labels, regression_targets,
                                                     proposals, matched_gt, self.box_coder,
                                                     giou_weight=self.giou_weight)
            losses = {'loss_classifier': loss_classifier, 'loss_box_reg': loss_box_reg}
        else:
            all_boxes, all_scores, all_labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes)
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
                 giou_weight=0.0):
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
        matched = [matcher(box_iou(g.to(p.dtype), p)) for p, g in zip(proposals_with_gt, gt_boxes)]
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

    def forward(self, features, proposals, image_shapes, targets=None):
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
                    class_logits, box_regression, boxes, image_shapes)
                detections = [{'boxes': b, 'labels': l, 'scores': s}
                              for b, s, l in zip(all_boxes, all_scores, all_labels)]
        return detections, losses
