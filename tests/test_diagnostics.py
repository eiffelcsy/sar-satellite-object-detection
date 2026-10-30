import torch

from sarbench.diagnostics import match_records, summarize


def test_records_split_lost_boxes_between_the_rpn_and_the_roi_head():
    gt = torch.tensor([[0., 0., 10., 10.], [50., 50., 60., 60.], [100., 100., 110., 110.]])
    labels = torch.tensor([3, 3, 7])
    # box 0: proposed and detected as its class; box 1: proposed, detected only as another class; box 2: never proposed
    proposals = torch.tensor([[0., 0., 10., 10.], [51., 50., 61., 60.], [300., 300., 310., 310.]])
    det_boxes = torch.tensor([[0., 0., 10., 11.], [51., 50., 61., 60.]])
    det_labels, det_scores = torch.tensor([3, 7]), torch.tensor([0.9, 0.4])
    records = match_records(gt, labels, proposals, det_boxes, det_labels, det_scores, top_k=1, image_id=5)
    assert [r['proposal_iou'] >= 0.5 for r in records] == [True, True, False]
    assert [r['top_k_proposal_iou'] >= 0.5 for r in records] == [True, False, False]  # only the first proposal
    assert [r['detection_iou'] >= 0.5 for r in records] == [True, True, False]
    assert [r['class_detection_iou'] >= 0.5 for r in records] == [True, False, False]
    assert records[0]['class_detection_score'] == torch.tensor(0.9).item() and records[0]['image_id'] == 5
    rows = summarize(records, {3: 'bridge', 7: 'playground'})
    bridge, playground = rows['bridge'], rows['playground']
    assert bridge['gt'] == 2 and bridge['proposal_recall'] == 1.0 and bridge['detection_recall'] == 0.5
    assert bridge['lost_in_rpn'] == 0.0 and bridge['lost_in_roi_head'] == 0.5
    assert playground['lost_in_rpn'] == 1.0 and playground['detection_recall'] == 0.0


def test_records_handle_images_without_detections_or_proposals():
    gt, labels = torch.tensor([[0., 0., 10., 10.]]), torch.tensor([1])
    empty = torch.zeros(0, 4)
    records = match_records(gt, labels, empty, empty, torch.zeros(0, dtype=torch.long), torch.zeros(0))
    assert records[0]['proposal_iou'] == 0.0 and records[0]['class_detection_score'] == 0.0
