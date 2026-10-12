"""Copy-paste, mosaic, scale-jitter and rotation augmentation helpers (CPU only)."""
import torch

from sarbench.data import _max_iou, copy_paste, drop_degenerate_boxes, mosaic_tiles, rotate90, scale_jitter


def test_drop_degenerate_boxes_removes_zero_area():
    boxes = torch.tensor([[0., 0., 10., 10.], [5., 5., 5., 20.], [1., 2., 9., 2.], [3., 3., 4., 4.]])
    labels = torch.tensor([1, 2, 3, 4])
    kept_boxes, kept_labels = drop_degenerate_boxes(boxes, labels)
    assert kept_labels.tolist() == [1, 4]  # the line (zero width) and zero height boxes are gone
    assert (kept_boxes[:, 2] - kept_boxes[:, 0] > 0).all() and (kept_boxes[:, 3] - kept_boxes[:, 1] > 0).all()


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


def test_scale_jitter_shrinking_pads_and_keeps_every_box():
    image = torch.rand(1, 32, 32)
    boxes = torch.tensor([[8., 8., 24., 24.]])
    out_image, out_boxes, out_labels = scale_jitter(image, boxes, torch.tensor([1]), (0.5, 0.5), size=32)
    assert out_image.shape == (1, 32, 32) and out_labels.tolist() == [1]
    assert (out_boxes[:, 2] - out_boxes[:, 0] > 0).all()  # nothing was removed


def test_scale_jitter_enlarging_keeps_a_box_on_canvas():
    image = torch.zeros(1, 32, 32)
    image[:, 12:20, 12:20] = 1.0
    boxes = torch.tensor([[12., 12., 20., 20.]])
    out_image, out_boxes, out_labels = scale_jitter(image, boxes, torch.tensor([2]), (1.2, 1.2), size=32)
    assert out_image.shape == (1, 32, 32) and len(out_boxes) >= 1
    assert out_boxes.min() >= 0 and out_boxes.max() <= 32


def test_copy_paste_keeps_the_signature_margin_around_the_box():
    size = 32
    image, boxes, labels = torch.zeros(1, size, size), torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)
    donor_image = torch.zeros(1, size, size)
    donor_image[:, 8:24, 8:24] = 1.0  # the object plus a 4 px halo (shadow / smearing)
    out_image, out_boxes, out_labels = copy_paste(
        image, boxes, labels, donor_image, torch.tensor([[12., 12., 20., 20.]]), torch.tensor([3]),
        max_iou=1.0, margin=4)
    assert len(out_boxes) == 1 and out_labels.tolist() == [3]
    x1, y1, x2, y2 = out_boxes[0].long().tolist()
    assert x2 - x1 == 8  # the pasted box hugs the object, not the margin-padded crop
    assert out_image[0, y2 + 1, x1 + 1] == 1.0  # halo outside the box was pasted too


def test_rotate90_moves_boxes_exactly_with_the_pixels():
    image = torch.zeros(1, 32, 32)
    image[:, 4:10, 2:20] = 1.0  # a bright rectangle filling exactly box [2, 4, 20, 10]
    boxes = torch.tensor([[2., 4., 20., 10.]])
    for k in range(4):
        out, out_boxes = rotate90(image, boxes, k)
        ys, xs = torch.nonzero(out[0] > 0.5, as_tuple=True)
        bright = torch.stack([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]).float()  # pixel i spans [i, i + 1)
        assert torch.equal(bright, out_boxes[0]), k
    assert torch.equal(rotate90(image, boxes, 4)[1], boxes)  # four quarter turns are the identity
