"""The Deformable-DETR head: training losses, Hungarian matching and eval detections (CPU only)."""
import torch

from sarbench.detr import DeformableDetrHead, MSDeformAttn


def tiny_head():
    return DeformableDetrHead(in_channels=16, num_classes=3, num_queries=10, level_names=('0', '1'),
                              d_model=32, n_heads=4, n_points=2, enc_layers=1, dec_layers=2,
                              dim_feedforward=64, dropout=0.0)


FEATURES = {'0': torch.randn(2, 16, 16, 16), '1': torch.randn(2, 16, 8, 8)}
TARGETS = [
    {'boxes': torch.tensor([[10., 12., 28., 40.], [2., 3., 9., 11.]]), 'labels': torch.tensor([1, 3])},
    {'boxes': torch.tensor([[30., 30., 50., 60.]]), 'labels': torch.tensor([2])},
]


def test_training_returns_finite_losses_and_backpropagates():
    head = tiny_head().train()
    _, losses = head(FEATURES, TARGETS, image_size=(64, 64))
    assert set(losses) == {'loss_classifier', 'loss_bbox', 'loss_giou', 'loss_aux'}
    assert all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
    assert all(p.grad is not None for p in head.parameters())


def test_eval_returns_detections_in_input_pixels():
    head = tiny_head().eval()
    with torch.no_grad():
        detections, losses = head(FEATURES, image_size=(64, 64))
    assert losses == {} and len(detections) == 2
    for det in detections:
        assert set(det) == {'boxes', 'labels', 'scores'}
        if len(det['boxes']):
            assert det['boxes'].shape[1] == 4
            assert det['boxes'].min() >= 0 and det['boxes'].max() <= 64
            assert det['labels'].min() >= 1 and det['labels'].max() <= 3
            assert det['scores'].max() <= 1 and det['scores'].min() >= 0


def test_empty_targets_are_handled():
    head = tiny_head().train()
    targets = [{'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)} for _ in range(2)]
    _, losses = head(FEATURES, targets, image_size=(64, 64))
    assert all(torch.isfinite(v) and v.item() >= 0 for v in losses.values())


def test_grad_checkpointing_matches_plain_and_backpropagates():
    def run(grad_checkpointing):
        torch.manual_seed(0)
        head = DeformableDetrHead(in_channels=16, num_classes=3, num_queries=10, level_names=('0', '1'),
                                  d_model=32, n_heads=4, n_points=2, enc_layers=2, dec_layers=2,
                                  dim_feedforward=64, dropout=0.0, grad_checkpointing=grad_checkpointing).train()
        _, losses = head(FEATURES, TARGETS, image_size=(64, 64))
        total = sum(losses.values())
        total.backward()
        return float(total), sum(p.grad is not None for p in head.parameters())

    plain, plain_grads = run(False)
    ckpt, ckpt_grads = run(True)
    assert abs(plain - ckpt) < 1e-5 and plain_grads == ckpt_grads


def test_msdeformattn_preserves_shape_and_differentiates():
    attn = MSDeformAttn(d_model=32, n_levels=2, n_heads=4, n_points=2)
    query = torch.randn(2, 5, 32, requires_grad=True)
    value = torch.randn(2, 16 * 16 + 8 * 8, 32)
    shapes = torch.tensor([[16, 16], [8, 8]])
    reference = torch.rand(2, 5, 2, 2)
    out = attn(query, reference, value, shapes, torch.tensor([0, 256]))
    assert out.shape == (2, 5, 32)
    out.sum().backward()
    assert query.grad is not None and attn.sampling_offsets.bias.grad is not None
