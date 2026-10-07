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


def test_dora_config_with_augmentation_stem_and_long_schedule():
    cfg = load_config(CONFIGS / 'dinov3_dora.yaml')
    assert cfg['backbone'] == 'dinov3' and cfg['adapt'] == 'dora'
    assert cfg['detail_stem'] is True
    assert cfg['mosaic'] > 0 and cfg['copy_paste'] > 0
    assert cfg['epochs'] == 48


def test_peft_configs_match_for_the_ablation():
    """dinov3_{dora,moelora,moedora} differ only in the PEFT method, so any metric gap is the adapter."""
    dora = load_config(CONFIGS / 'dinov3_dora.yaml')
    shared = ('backbone', 'init', 'peft_targets', 'detail_stem', 'pseudo_rgb', 'edge',
              'mosaic', 'copy_paste', 'loss_weights', 'epochs', 'preprocess_cache')
    for name in ('moelora', 'moedora'):
        cfg = load_config(CONFIGS / f'dinov3_{name}.yaml')
        assert all(dora[key] == cfg[key] for key in shared), name
        assert dora['backbone_kwargs']['model_name'] == cfg['backbone_kwargs']['model_name']
        assert cfg['adapt'] == name
        assert cfg['peft_kwargs']['experts'] * cfg['peft_kwargs']['rank'] == dora['peft_kwargs']['rank']
        assert cfg['peft_kwargs']['alpha'] / cfg['peft_kwargs']['rank'] \
            == dora['peft_kwargs']['alpha'] / dora['peft_kwargs']['rank']


def test_loss_weights_default_empty_and_parse_from_config():
    assert load_config(CONFIGS / 'vit_lora.yaml')['loss_weights'] == {}
    weights = load_config(CONFIGS / 'dinov3_dora.yaml')['loss_weights']
    assert weights['classification'] == 1.0
    assert weights['loss_box_reg'] > 1.0  # the config upweights the RoI box loss


def test_every_config_loads():
    paths = sorted(CONFIGS.glob('*.yaml'))
    assert paths, 'no configs/ found'
    for path in paths:
        load_config(path)
