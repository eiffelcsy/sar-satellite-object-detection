"""The Deformable-Conv RoI head and Cascade R-CNN (CPU only)."""
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.ops import MultiScaleRoIAlign

from sarbench.heads import CascadeRoIHeads, DeformConvBoxHead


def _features():
    return {name: torch.randn(2, 256, size, size) for name, size in zip(['0', '1', '2', '3'], [32, 16, 8, 4])}


def _proposals():
    return [torch.cat([torch.rand(20, 4) * 100, torch.tensor([[0., 0., 10., 10.]])]) for _ in range(2)]


TARGETS = [{'boxes': torch.tensor([[10., 10., 30., 30.]]), 'labels': torch.tensor([1])},
           {'boxes': torch.tensor([[40., 40., 80., 90.]]), 'labels': torch.tensor([2])}]
SHAPES = [(128, 128), (128, 128)]


def test_deform_conv_box_head_output_dim():
    assert DeformConvBoxHead()(torch.randn(3, 256, 7, 7)).shape == (3, 1024)


def test_cascade_roi_heads_train_and_eval():
    pool = MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    heads = [TwoMLPHead(256 * 7 * 7, 1024)]  # shared box head; one predictor per stage
    predictors = [FastRCNNPredictor(1024, 10) for _ in range(3)]
    cascade = CascadeRoIHeads(pool, heads, predictors)

    cascade.train()
    detections, losses = cascade(_features(), _proposals(), SHAPES, TARGETS)
    assert detections == [] and set(losses) == {'loss_classifier', 'loss_box_reg'}
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert any(p.grad is not None for p in cascade.parameters())

    cascade.eval()
    with torch.no_grad():
        detections, losses = cascade(_features(), _proposals(), SHAPES)
    assert losses == {} and len(detections) == 2
    if len(detections[0]['labels']):
        assert detections[0]['labels'].dtype == torch.int64 and detections[0]['labels'].min() >= 1
