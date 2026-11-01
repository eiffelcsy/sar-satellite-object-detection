"""rfdetr_track/fusion.py: the P3 weight remap, the image-class targets, and that the DINOv3 branch is a no-op at
init. Needs rfdetr (the .venv-rfdetr environment); the models are randomly initialised, so nothing is downloaded
except a DINOv3 config (that test skips without it)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip('rfdetr')
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rfdetr_track'))
import fusion  # noqa: E402

DINO_CONFIG = 'Tooony133/dinov3-vits16-pretrain-lvd1689m'  # a ViT-S/16 config, public (the weights are not needed)


def large(**kwargs):
    import rfdetr
    return rfdetr.RFDETRLarge(pretrain_weights=None, num_classes=9, resolution=512, **kwargs).model.model


def test_p4_checkpoint_remaps_into_a_p3_model():
    fusion.install(None)
    source, target = large().state_dict(), large(projector_scale=['P3', 'P4'])
    state = fusion.remap_p3(target, source)
    target.load_state_dict(state, strict=False)  # raises on any shape mismatch left
    loaded = target.state_dict()
    old, new = 'backbone.0.projector.stages.0.', 'backbone.0.projector.stages.1.'
    assert all(torch.equal(loaded[new + k[len(old):]], v) for k, v in source.items() if k.startswith(old))
    attn = 'transformer.decoder.layers.0.cross_attn.attention_weights.bias'
    heads, points = 16, 2
    p3, p4 = loaded[attn].view(heads, 2, points).unbind(1)
    assert torch.equal(p4, source[attn].view(heads, points))
    assert torch.allclose(p3, p4 + fusion.P3_ATTENTION_OFFSET)
    assert fusion.remap_p3(target, loaded) is loaded  # a P3 checkpoint is left alone


def test_image_labels_take_the_most_common_box_class():
    targets = [{'labels': torch.tensor([2, 2, 5])}, {'labels': torch.tensor([], dtype=torch.long)}]
    assert fusion.image_labels(targets).tolist() == [2, -1]
    logits = torch.zeros(2, 9, requires_grad=True)
    assert fusion.image_cls_loss(logits, [targets[1], targets[1]]).item() == 0  # no labelled image: zero, no nan


def test_dino_branch_is_a_no_op_at_init():
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list
    spec = {'dino': {'model_name': DINO_CONFIG, 'pretrained': False, 'dora_rank': 4, 'convpass_dim': 16},
            'image_cls_weight': 0.1}
    try:
        fusion.install(spec)
        fused = large()
    except OSError:
        pytest.skip(f'no {DINO_CONFIG} config (offline)')
    finally:
        fusion.install(None)
    plain = large()
    plain.load_state_dict({k: v for k, v in fused.state_dict().items() if k in plain.state_dict()})
    images = nested_tensor_from_tensor_list(list(torch.rand(2, 3, 512, 512)))
    fused.eval(), plain.eval()
    with torch.no_grad():
        a, b = fused(images), plain(images)
    assert torch.allclose(a['pred_boxes'], b['pred_boxes'], atol=1e-5)
    assert torch.allclose(a['pred_logits'], b['pred_logits'], atol=1e-4)
    assert a['pred_image_logits'].shape == (2, 9)
