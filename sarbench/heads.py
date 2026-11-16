"""The RoI head: torchvision's Faster R-CNN head with a GIoU term added to its box-regression loss."""
import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import generalized_box_iou


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets, proposals, matched_gt, box_coder,
                  giou_weight=1.0):
    """torchvision's Fast R-CNN loss (cross-entropy + smooth-L1) plus `giou_weight` x the GIoU loss of each positive's
    decoded box (its GT class) against the GT box it was matched to, normalized like the smooth-L1 term."""
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    num_classes = class_logits.shape[1]
    positive = torch.where(labels > 0)[0]
    box_regression = box_regression.reshape(box_regression.size(0), num_classes, 4)
    box_loss = F.smooth_l1_loss(box_regression[positive, labels[positive]], regression_targets[positive],
                                beta=1 / 9, reduction='sum') / labels.numel()

    if giou_weight > 0:
        counts = [len(p) for p in proposals]
        decoded = box_coder.decode(box_regression.flatten(1), proposals).split(counts, dim=0)
        terms = []
        for predicted, gt, label in zip(decoded, matched_gt, labels.split(counts)):
            keep = label > 0
            if not keep.any():
                continue
            predicted = predicted.view(-1, num_classes, 4)[keep]
            predicted = predicted[torch.arange(predicted.shape[0], device=predicted.device), label[keep]]
            terms.append((1 - torch.diag(generalized_box_iou(predicted, gt[keep]))).sum())
        if terms:
            box_loss = box_loss + giou_weight * (sum(terms) / labels.numel())
    return classification_loss, box_loss


class DetRoIHeads(RoIHeads):
    """torchvision's RoI head (box branch only) with the GIoU box term (`giou_weight`). Returns torchvision's
    (per-image detections, loss dict)."""

    def __init__(self, *args, giou_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.giou_weight = giou_weight

    def forward(self, features, proposals, image_shapes, targets=None):
        matched_gt = None
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            # the GT box matched to each sampled proposal (an image may have no boxes left after augmentation)
            matched_gt = [t['boxes'][mi.clamp(min=0)].to(proposals[0].dtype) if len(t['boxes'])
                          else proposals[0].new_zeros((0, 4)) for t, mi in zip(targets, matched_idxs)]
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets,
                                                          proposals, matched_gt, self.box_coder, self.giou_weight)
            return [], {'loss_classifier': loss_classifier, 'loss_box_reg': loss_box_reg}
        boxes, scores, classes = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        return [{'boxes': b, 'labels': c, 'scores': s} for b, s, c in zip(boxes, scores, classes)], {}
