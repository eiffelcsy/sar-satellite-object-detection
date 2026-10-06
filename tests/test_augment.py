"""Copy-paste and mosaic augmentation helpers (CPU only)."""
import torch

from sarbench.data import _max_iou, copy_paste, mosaic_tiles


def test_mosaic_tiles_shape_boxes_and_labels():
    size = 32
    tiles = [(torch.rand(1, size, size), torch.tensor([[4., 4., 12., 12.], [20., 20., 28., 28.]]),
              torch.tensor([1, 2])) for _ in range(4)]
    image, boxes, labels = mosaic_tiles(tiles, size)
    assert image.shape == (1, size, size)
    assert len(boxes) == 8 and len(labels) == 8
    assert boxes.min() >= 0 and boxes.max() <= size
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()


def test_copy_paste_adds_a_box_and_pastes_pixels():
    size = 32
    image = torch.zeros(1, size, size)
    boxes = torch.tensor([[0., 0., 6., 6.]])
    labels = torch.tensor([1])
    donor_image = torch.ones(1, size, size)
    out_image, out_boxes, out_labels = copy_paste(
        image, boxes, labels, donor_image, torch.tensor([[8., 8., 20., 24.]]), torch.tensor([3]), max_iou=1.0)
    assert len(out_boxes) == 2 and out_labels.tolist() == [1, 3]
    assert out_boxes.min() >= 0 and out_boxes.max() <= size
    assert out_image.max() == 1.0  # the donor crop (all ones) was pasted somewhere


def test_max_iou():
    boxes = torch.tensor([[0., 0., 10., 10.]])
    assert _max_iou(torch.tensor([0., 0., 10., 10.]), boxes) == 1.0
    assert _max_iou(torch.tensor([20., 20., 30., 30.]), boxes) == 0.0
