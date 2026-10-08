"""Alternative RoI detection heads for the ablation: a Deformable-Conv box head and Cascade R-CNN.

Both plug into torchvision's Faster R-CNN RoI head and return the same `(per-image detections, loss dict)`
interface as `RoIHeads`, so the rest of the pipeline is unchanged.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection._utils import BalancedPositiveNegativeSampler, Matcher
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import DeformConv2d, box_iou, clip_boxes_to_image


class DeformConvBoxHead(nn.Module):
    """RoI box head using deformable convolutions (Dai et al., 2017) on the pooled 256 x 7 x 7 features.

    Small objects land in few, misaligned RoI cells; the learned offsets sample the feature map around the true
    object shape instead of a fixed grid. The 1024-d output matches `FastRCNNPredictor`.
    """

    def __init__(self, in_channels=256, out_channels=1024, kernel_size=3, num_convs=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_convs):
            self.blocks.append(nn.ModuleDict({
                'conv': nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2),
                'offset': nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size,
                                    padding=kernel_size // 2),
                'dcn': DeformConv2d(in_channels, in_channels, kernel_size, padding=kernel_size // 2)}))
        self.fc = nn.Linear(in_channels * 7 * 7, out_channels)

    def forward(self, x):
        for block in self.blocks:
            x = F.relu(block['dcn'](F.relu(block['conv'](x)), block['offset'](x)))
        return F.relu(self.fc(x.flatten(1)))


class CascadeRoIHeads(RoIHeads):
    """Cascade R-CNN (Cai & Vasconcelos, 2018): S RoI stages with increasing IoU thresholds, each decoding
    boxes that are re-pooled and refined by the next stage. Reuses torchvision's box coder and post-processing;
    the per-stage losses are summed into `loss_classifier` / `loss_box_reg` so the `loss:` weights still apply.
    """

    def __init__(self, box_roi_pool, box_heads, box_predictors, fg_iou_thresholds=(0.5, 0.6, 0.7),
                 bg_iou_threshold=0.5, batch_size_per_image=512, positive_fraction=0.25, bbox_reg_weights=None,
                 score_thresh=0.05, nms_thresh=0.5, detections_per_img=100, stage_loss_weights=None):
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
        matched_gt = [g[cm] for g, cm in zip(gt_boxes, sampled_clamped)]
        regression_targets = self.box_coder.encode(matched_gt, sampled_boxes)
        return sampled_boxes, sampled_labels, regression_targets

    def forward(self, features, proposals, image_shapes, targets=None):
        boxes, labels, regression_targets = proposals, None, None
        if self.training:
            boxes, labels, regression_targets = self._sample(
                proposals, targets, self.cascade_matchers[0], self.cascade_samplers[0])
        losses = {}
        detections = []
        for stage, predictor in enumerate(self.box_predictors):
            head = self.box_heads[min(stage, len(self.box_heads) - 1)]  # shared head unless per-stage provided
            if stage > 0 and self.training:
                boxes, labels, regression_targets = self._sample(
                    boxes, targets, self.cascade_matchers[stage], self.cascade_samplers[stage])
            # At inference the refined boxes from the previous stage carry over as this stage's proposals.
            pooled = self.box_roi_pool(features, boxes, image_shapes)
            class_logits, box_regression = predictor(head(pooled))
            if self.training:
                loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels,
                                                              regression_targets)
                weight = self.stage_loss_weights[stage]
                losses['loss_classifier'] = losses.get('loss_classifier', 0.) + weight * loss_classifier
                losses['loss_box_reg'] = losses.get('loss_box_reg', 0.) + weight * loss_box_reg
                boxes = self.box_coder.decode(box_regression.detach(), boxes)
                boxes = [clip_boxes_to_image(b, s) for b, s in zip(boxes, image_shapes)]
            elif stage < len(self.box_predictors) - 1:
                # Keep every refined proposal between stages (no score threshold / NMS): the later stages need
                # high-recall candidates for small objects. Only the final stage thresholds and suppresses.
                refined = self.box_coder.decode(box_regression, boxes)
                boxes = [clip_boxes_to_image(b, s) for b, s in zip(refined, image_shapes)]
            else:
                all_boxes, all_scores, all_labels = self.postprocess_detections(
                    class_logits, box_regression, boxes, image_shapes)
                detections = [{'boxes': b, 'labels': l, 'scores': s}
                              for b, s, l in zip(all_boxes, all_scores, all_labels)]
        return detections, losses
