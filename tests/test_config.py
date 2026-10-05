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


def test_every_config_loads():
    paths = sorted(CONFIGS.glob('*.yaml'))
    assert paths, 'no configs/ found'
    for path in paths:
        load_config(path)
