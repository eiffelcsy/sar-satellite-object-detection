"""The YAML run configs: backbone / PEFT mapping, fallbacks and defaults (no GPU needed)."""
from pathlib import Path

from sarbench.config import DEFAULTS, load_config

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def test_vit_lora_config():
    cfg = load_config(CONFIGS / 'vit_lora.yaml')
    assert cfg['backbone'] == 'vit' and cfg['init'] == 'pretrained' and cfg['adapt'] == 'lora'
    assert cfg['backbone_kwargs']['timm_name'].startswith('vit_base')
    assert cfg['peft_kwargs'] == {'rank': 16, 'alpha': 32}
    assert cfg['epochs'] == DEFAULTS['epochs']  # absent train section falls back to the reference protocol


def test_full_pretrained_fine_tunes_gently():
    cfg = load_config(CONFIGS / 'vit_full.yaml')
    assert cfg['adapt'] == 'full' and cfg['backbone_lr'] == 2e-5


def test_scratch_config():
    cfg = load_config(CONFIGS / 'terramind_scratch.yaml')
    assert cfg['backbone'] == 'terramind' and cfg['init'] == 'scratch' and cfg['adapt'] == 'full'


def test_dinov3_config_selects_the_checkpoint_and_lora_location():
    cfg = load_config(CONFIGS / 'dinov3_lora.yaml')
    assert cfg['backbone'] == 'dinov3'
    assert cfg['backbone_kwargs']['model_name'].startswith('facebook/dinov3')
    assert cfg['peft_targets'] == ['attn', 'mlp']  # q/k/v + o_proj and the MLP fc1/fc2
    assert cfg['peft_kwargs'] == {'rank': 32, 'alpha': 32}
    assert cfg['pseudo_rgb'] is True and cfg['edge'] == 'sobel'  # 1-channel SAR -> 3-channel DINOv3 input


def test_detector_defaults_to_faster_rcnn_and_can_be_switched():
    assert load_config(CONFIGS / 'vit_lora.yaml')['detector'] == 'faster_rcnn'
    cfg = load_config(CONFIGS / 'dinov3_deformable_detr_lora.yaml')
    assert cfg['detector'] == 'deformable_detr'
    assert cfg['detector_kwargs']['num_queries'] == 300
    assert cfg['detector_kwargs']['level_names'] == ['0', '1', '2', '3']


def test_loss_weights_default_empty_and_parse_from_config():
    assert load_config(CONFIGS / 'vit_lora.yaml')['loss_weights'] == {}
    weights = load_config(CONFIGS / 'dinov3_deformable_detr_lora.yaml')['loss_weights']
    assert weights['classification'] == 1.0
    assert weights['loss_bbox'] > 1.0 and weights['loss_giou'] > 1.0  # the config upweights the box terms


def test_every_config_loads():
    paths = sorted(CONFIGS.glob('*.yaml'))
    assert paths, 'no configs/ found'
    for path in paths:
        load_config(path)
