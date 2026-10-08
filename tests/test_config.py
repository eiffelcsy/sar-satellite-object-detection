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
    """dinov3_{dora,moelora} differ only in the PEFT method, so any metric gap is the adapter."""
    dora = load_config(CONFIGS / 'dinov3_dora.yaml')
    shared = ('backbone', 'init', 'peft_targets', 'detail_stem', 'pseudo_rgb', 'edge',
              'mosaic', 'copy_paste', 'loss_weights', 'epochs', 'preprocess_cache')
    for name in ('moelora',):
        cfg = load_config(CONFIGS / f'dinov3_{name}.yaml')
        assert all(dora[key] == cfg[key] for key in shared), name
        assert dora['backbone_kwargs']['model_name'] == cfg['backbone_kwargs']['model_name']
        assert cfg['adapt'] == name
        assert cfg['peft_kwargs']['experts'] * cfg['peft_kwargs']['rank'] == dora['peft_kwargs']['rank']
        assert cfg['peft_kwargs']['alpha'] / cfg['peft_kwargs']['rank'] \
            == dora['peft_kwargs']['alpha'] / dora['peft_kwargs']['rank']


def test_splus_ablation_only_changes_the_backbone_checkpoint():
    dora = load_config(CONFIGS / 'dinov3_dora.yaml')
    splus = load_config(CONFIGS / 'dinov3_splus_dora.yaml')
    assert splus['backbone'] == dora['backbone'] == 'dinov3'
    assert splus['backbone_kwargs']['model_name'] == 'facebook/dinov3-vits16plus-pretrain-lvd1689m'
    for key in ('adapt', 'peft_kwargs', 'peft_targets', 'detail_stem', 'pseudo_rgb', 'edge',
                'mosaic', 'copy_paste', 'loss_weights', 'epochs'):
        assert splus[key] == dora[key], key


def test_head_ablation_configs_match_the_baseline_and_add_fusion():
    base = load_config(CONFIGS / 'dinov3_dora.yaml')
    shared = ('backbone', 'init', 'backbone_kwargs', 'adapt', 'peft_kwargs', 'peft_targets', 'detail_stem',
              'pseudo_rgb', 'edge', 'mosaic', 'copy_paste', 'loss_weights', 'epochs', 'preprocess_cache')
    assert base['fusion_layers'] is None and base['head'] == 'standard'
    deform = load_config(CONFIGS / 'dinov3_dora_deform.yaml')
    assert deform['head'] == 'deform' and deform['fusion_layers'] == [5, 8, 11]
    assert all(deform[key] == base[key] for key in shared)
    cascade = load_config(CONFIGS / 'dinov3_dora_cascade.yaml')
    assert cascade['head'] == 'cascade' and cascade['fusion_layers'] == [5, 8, 11]
    assert cascade['head_params']['num_stages'] == 3
    assert all(cascade[key] == base[key] for key in shared)


def test_detection_loss_ablation_configs():
    base = load_config(CONFIGS / 'dinov3_dora.yaml')
    assert (base['giou_weight'], base['focal_loss'], base['roi_sampling_ratio']) == (0.0, False, 2)
    shared = ('backbone', 'init', 'backbone_kwargs', 'adapt', 'peft_kwargs', 'peft_targets', 'detail_stem',
              'pseudo_rgb', 'edge', 'mosaic', 'copy_paste', 'loss_weights', 'epochs', 'preprocess_cache')
    for name, giou, focal in (('giou', 1.0, False), ('focal', 0.0, True), ('giou_focal', 1.0, True)):
        cfg = load_config(CONFIGS / f'dinov3_dora_{name}.yaml')
        assert cfg['giou_weight'] == giou and cfg['focal_loss'] is focal
        assert all(cfg[key] == base[key] for key in shared), name


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
