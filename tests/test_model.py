import torch
from torch import nn

from sarbench.adapters import ConvAdapter, DoRA
from sarbench.config import load_config
from sarbench.heads import fastrcnn_loss
from sarbench.model import MAX_TOTAL_PARAMETERS, MAX_TRAINABLE_PARAMETERS, build_model, parameter_counts
from train import ROOT

CONFIG = load_config(ROOT / 'configs' / 'final.yaml')


def test_adapters_start_as_noops():
    torch.manual_seed(0)
    base, x = nn.Linear(16, 8), torch.randn(4, 16)
    assert torch.allclose(DoRA(base, rank=4)(x), base(x), atol=1e-6)
    tokens = torch.randn(2, 1 + 16, 32)  # 1 prefix token + a 4 x 4 grid
    assert torch.equal(ConvAdapter(32, 8, num_prefix_tokens=1)(tokens), tokens)


def test_giou_term_vanishes_for_a_perfect_box():
    from torchvision.models.detection._utils import BoxCoder
    coder = BoxCoder((10., 10., 5., 5.))
    proposals, gt = [torch.tensor([[10., 10., 30., 30.]])], [torch.tensor([[10., 10., 30., 30.]])]
    regression = torch.zeros(1, 4 * 3)
    args = (torch.tensor([[0., 9., 0.]]), regression, [torch.tensor([1])], [torch.zeros(1, 4)], proposals, gt, coder)
    _, with_giou = fastrcnn_loss(*args, giou_weight=1.0)
    _, without = fastrcnn_loss(*args, giou_weight=0.0)
    assert with_giou.item() == without.item() == 0


def test_the_final_model_has_the_reported_parameter_counts():
    model, pretrained_ids = build_model(CONFIG, pretrained=False)  # the real architecture, random weights
    counts = parameter_counts(model, pretrained_ids)
    assert (counts['total'], counts['trainable']) == (123_932_325, 38_862_501)
    assert counts['total'] <= MAX_TOTAL_PARAMETERS and counts['trainable'] <= MAX_TRAINABLE_PARAMETERS


def test_training_and_inference_forward(tiny_dinov3):
    torch.manual_seed(0)
    model, _ = build_model(CONFIG, pretrained=False)
    images = torch.rand(2, 3, 512, 512)
    targets = [{'boxes': torch.tensor([[100., 100., 120., 116.]]), 'labels': torch.tensor([3])}] * 2
    logits, detections, losses = model.train()(images, targets)
    assert logits.shape == (2, 9) and detections == []
    assert set(losses) == {'loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg'}
    assert all(loss.isfinite() for loss in losses.values())
    with torch.no_grad():
        _, detections, losses = model.eval()(images)
    assert losses == {} and len(detections) == 2
    for d in detections:
        assert len(d['boxes']) <= 100 and d['boxes'].dtype == torch.float32 and d['labels'].min() >= 1
