"""The RoI detection loss: torchvision's Fast R-CNN loss plus an optional GIoU box-regression term."""
import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets, proposals=None, matched_gt=None,
                  box_coder=None, giou_weight=0.0):
    """torchvision's RoI loss with an additional GIoU box term.

    `proposals` (sampled boxes, one tensor per image) and `matched_gt` (the GT box matched to each proposal) are
    only needed when `giou_weight > 0`.
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    num_classes = class_logits.shape[1]
    positive = torch.where(labels > 0)[0]
    labels_pos = labels[positive]
    box_regression = box_regression.reshape(box_regression.size(0), num_classes, 4)
    box_loss = F.smooth_l1_loss(box_regression[positive, labels_pos], regression_targets[positive],
                                beta=1 / 9, reduction='sum') / labels.numel()

    if giou_weight > 0 and proposals is not None and matched_gt is not None:
        counts = [len(p) for p in proposals]
        # BoxCoder.decode returns one concatenated tensor; split it back per image.
        decoded = box_coder.decode(box_regression.flatten(1), proposals).split(counts, dim=0)
        terms = []
        for predicted, gt, label in zip(decoded, matched_gt, labels.split(counts)):
            keep = label > 0
            if not keep.any():
                continue
            predicted = predicted.view(-1, num_classes, 4)[keep]
            matched_class = label[keep]
            predicted = predicted[torch.arange(predicted.shape[0], device=predicted.device), matched_class]
            terms.append((1 - torch.diag(generalized_box_iou(predicted, gt[keep]))).sum())
        if terms:
            box_loss = box_loss + giou_weight * (sum(terms) / labels.numel())
    return classification_loss, box_loss
