"""Detection losses shared by every head: focal classification and a GIoU box-regression term.

`multiclass_focal_loss` is a focal-weighted cross-entropy (gamma = 0 gives plain CE); the RPN uses the binary
sigmoid-focal form on its objectness logits. `fastrcnn_loss` mirrors torchvision's RoI loss but adds the
optional focal classification and GIoU box term, so standard, Deformable-Conv and Cascade heads all share it.
"""
import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou


def binary_focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.25, reduction='mean'):
    """Sigmoid focal loss (Lin et al., 2017) on raw logits; used for the RPN objectness."""
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probability * targets + (1 - probability) * (1 - targets)
    loss = ce * (1 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    return loss.mean() if reduction == 'mean' else loss.sum()


def multiclass_focal_loss(logits, targets, gamma=2.0, weight=None, reduction='mean'):
    """Focal-weighted cross-entropy over `C` classes (gamma = 0 is exactly cross-entropy)."""
    log_p_t = F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    loss = -(1 - log_p_t.exp()).pow(gamma) * log_p_t
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean() if reduction == 'mean' else loss.sum()


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets, proposals=None, matched_gt=None,
                  box_coder=None, focal=False, focal_gamma=2.0, giou_weight=0.0):
    """torchvision's RoI loss with optional focal classification and an additional GIoU box term.

    `proposals` (sampled boxes, one tensor per image) and `matched_gt` (the GT box matched to each proposal) are
    only needed when `giou_weight > 0`.
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = multiclass_focal_loss(class_logits, labels, gamma=focal_gamma) if focal \
        else F.cross_entropy(class_logits, labels)

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
