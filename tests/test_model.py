"""sarbench/model.py on real images (GPU): outputs and losses, fp32 detection heads under bf16 autocast, and an
overfit check (the base recipe's DINOv3 + DoRA fits 16 training images). DINOv3 is gated on Hugging Face."""
from pathlib import Path

import torch
import torch.nn.functional as F

from sarbench.adapters import adapt_backbone
from sarbench.backbones import DINOv3
from sarbench.data import SARMultiTask, collate, to_original_xywh
from sarbench.metrics import detection_metrics
from sarbench.model import MultiTaskModel

ROOT = Path(__file__).resolve().parents[2] / 'dataset' / 'SARFact-Course-20K'


def first_train_images(n):
    """The first n training images (no augmentation), their labels and their targets, on the GPU."""
    dataset = SARMultiTask(ROOT, 'train', limit=n)
    images, labels, targets = collate([dataset[i] for i in range(n)])
    return images.cuda(), labels.cuda(), [{**t, 'boxes': t['boxes'].cuda(), 'labels': t['labels'].cuda()}
                                          for t in targets]


def forward(model, images, targets=None):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(images, targets)


def test_training_forward_returns_logits_and_the_four_losses():
    model = MultiTaskModel(DINOv3(pretrained=False)).cuda().train()
    images, _, targets = first_train_images(2)
    logits, detections, losses = forward(model, images, targets)
    assert logits.shape == (2, 9) and detections == []
    assert set(losses) == {'loss_objectness', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg'}
    assert all(loss.dtype == torch.float32 and loss.isfinite() for loss in losses.values())


def test_detection_heads_run_in_fp32_and_boxes_are_not_quantized():
    model = MultiTaskModel(DINOv3(pretrained=False)).cuda().eval()
    dtypes = set()  # of the RPN's and the box head's outputs: bf16 here would snap boxes to a 2 px grid near 512
    model.rpn.head.register_forward_hook(lambda module, inputs, out: dtypes.update(t.dtype for t in out[0] + out[1]))
    model.roi_heads.box_predictor.register_forward_hook(lambda module, inputs, out: dtypes.update(t.dtype for t in out))
    with torch.no_grad():
        _, detections, losses = forward(model, first_train_images(2)[0])
    assert dtypes == {torch.float32} and losses == {}
    boxes, labels = torch.cat([d['boxes'] for d in detections]), torch.cat([d['labels'] for d in detections])
    assert len(boxes) > 0 and boxes.dtype == torch.float32 and detections[0]['scores'].dtype == torch.float32
    assert boxes.min() >= 0 and boxes.max() <= 512 and labels.min() >= 1 and labels.max() <= 9
    assert (boxes != boxes.bfloat16().float()).any()  # not every coordinate is on the bf16 grid


def test_dora_overfits_16_training_images():
    torch.manual_seed(0)
    images, labels, targets = first_train_images(16)
    backbone = DINOv3(pretrained=True)
    adapt_backbone(backbone)
    model = MultiTaskModel(backbone, detail_stem=True, giou_weight=1.0).cuda().train()
    heads = [p for name, p in model.named_parameters() if not name.startswith('backbone.')]
    optimizer = torch.optim.AdamW([{'params': [p for p in model.backbone.parameters() if p.requires_grad]},
                                   {'params': heads}], lr=1e-4)
    for _ in range(200):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _, losses = model(images, targets)
            loss = F.cross_entropy(logits, labels) + sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, detections, _ = forward(model.eval(), images)
    assert (logits.argmax(dim=1) == labels).all()  # accuracy 100 % (16 images miss 2 classes: no macro metrics)
    predictions = []
    for t, d in zip(targets, detections):
        boxes = to_original_xywh(d['boxes'], t['orig_size']).tolist()
        predictions += [{'image_id': t['image_id'], 'category_id': c, 'bbox': b, 'score': s}
                        for b, c, s in zip(boxes, d['labels'].tolist(), d['scores'].tolist())]
    image_ids = [t['image_id'] for t in targets]
    assert detection_metrics(ROOT / 'detection' / 'instances_train.json', predictions, image_ids)['AP50'] > 0.9
