"""Where ground-truth boxes are lost: never proposed by the RPN, or proposed but not kept by the RoI head.

`match_records` takes one image's GT boxes and the model's candidates at each stage and returns one record per GT
box: its best IoU with any RPN proposal (all of them, and the top `top_k` by objectness), with any final detection
(class-agnostic), and with a detection of its own class, plus that detection's score. `summarize` turns the records
into per-class recall at an IoU threshold. A GT box with a proposal but no detection is lost in the RoI head; one
without a proposal is lost in the RPN.
"""
from collections import defaultdict

import torch
from torchvision.ops import box_iou


def _best(gt, boxes):
    """Best IoU of each GT box with `boxes` and the index of that box (-1 when there are none)."""
    if len(gt) == 0 or len(boxes) == 0:
        return gt.new_zeros(len(gt)), torch.full((len(gt),), -1, dtype=torch.long, device=gt.device)
    return box_iou(gt, boxes).max(dim=1)


def match_records(gt_boxes, gt_labels, proposals, det_boxes, det_labels, det_scores, top_k=300, image_id=None):
    """One record per GT box (XYXY, same canvas for every input). `proposals` are in objectness order."""
    gt_boxes, proposals, det_boxes = gt_boxes.float(), proposals.float(), det_boxes.float()
    prop_iou, _ = _best(gt_boxes, proposals)
    top_iou, _ = _best(gt_boxes, proposals[:top_k])
    any_iou, _ = _best(gt_boxes, det_boxes)
    records = []
    for i, (box, label) in enumerate(zip(gt_boxes, gt_labels.tolist())):
        same = det_labels == label
        cls_iou, idx = _best(box[None], det_boxes[same])
        score = float(det_scores[same][idx[0]]) if idx[0] >= 0 else 0.0
        w, h = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        records.append({'image_id': image_id, 'label': label, 'size': (w * h) ** 0.5,
                        'aspect': max(w, h) / max(min(w, h), 1e-6),
                        'proposal_iou': float(prop_iou[i]), 'top_k_proposal_iou': float(top_iou[i]),
                        'detection_iou': float(any_iou[i]), 'class_detection_iou': float(cls_iou[0]),
                        'class_detection_score': score})
    return records


def summarize(records, names, iou=0.5):
    """Per class: GT count and the fraction of GT boxes reached at each stage at `iou`, plus where the rest are lost."""
    by_class = defaultdict(list)
    for r in records:
        by_class[r['label']].append(r)
    rows = {}
    for label in sorted(by_class):
        rs = by_class[label]
        n = len(rs)
        frac = lambda key: sum(r[key] >= iou for r in rs) / n
        found = [r['class_detection_score'] for r in rs if r['class_detection_iou'] >= iou]
        rows[names.get(label, str(label))] = {
            'gt': n, 'proposal_recall': frac('proposal_iou'), 'top_k_proposal_recall': frac('top_k_proposal_iou'),
            'detection_recall_any_class': frac('detection_iou'), 'detection_recall': frac('class_detection_iou'),
            'lost_in_rpn': 1 - frac('proposal_iou'),
            'lost_in_roi_head': frac('proposal_iou') - frac('class_detection_iou'),
            'median_score_found': sorted(found)[len(found) // 2] if found else 0.0}
    return rows
