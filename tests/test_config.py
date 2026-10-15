"""The YAML run configs: the base recipe, `base:` inheritance, one-factor ablations and the parameter budget."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sarbench.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
BASE = CONFIGS / 'dinov3_dora_giou.yaml'
ABLATIONS = sorted((CONFIGS / 'ablations').glob('*.yaml'))
# Keys that only name or describe a run; everything else is the recipe.
NON_RECIPE = {'name'}


def recipe_diff(a, b):
    return {key for key in a if key not in NON_RECIPE and a[key] != b[key]}


def test_base_is_the_dora_giou_recipe():
    cfg = load_config(BASE)
    assert cfg['backbone_kwargs']['model_name'] == 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    assert cfg['peft_kwargs'] == {'rank': 32, 'alpha': 32} and cfg['peft_targets'] == ['attn', 'mlp']
    assert cfg['giou_weight'] == 1.0 and cfg['detail_stem'] is True and cfg['head'] == 'standard'
    assert cfg['pseudo_rgb'] is True and (cfg['mosaic'], cfg['copy_paste']) == (0.3, 0.5)
    assert cfg['epochs'] == 48 and cfg['loss_weights']['loss_box_reg'] == 2.0
    assert cfg['copy_paste_margin'] == 0 and cfg['rotation90'] is False  # the original run's augmentation


@pytest.mark.parametrize('name, parent, changed', [
    ('a1_input', BASE, {'input_adapter'}),
    ('a2_aug', BASE, {'rotation90', 'scale_jitter', 'copy_paste_margin'}),
    ('a3_wst', BASE, {'wst_stem'}),
    ('a4_warmstart', BASE, {'stage_a_epochs'}),
    ('a5_fusion', BASE, {'fusion_layers'}),
    ('a6_warmstart_cascade', 'a4_warmstart', {'head', 'head_params'}),
    ('b1_full_blocks', 'a1_input', {'full_blocks'}),
    ('b2_rank_early', 'a1_input', {'rank_overrides'}),
    ('b3_rank_uniform', 'a1_input', {'peft_kwargs'}),
    ('b4_convpass', 'a1_input', {'conv_adapters'}),
    ('b5_hybrid', 'a1_input', {'full_blocks', 'rank_overrides', 'conv_adapters'}),
    ('b4_convpass_58ep', 'b4_convpass', {'epochs'}),
    ('c1a_b4_warmstart', 'b4_convpass', {'stage_a_epochs'}),
    ('c1b_b4_warmstart_cascade', 'c1a_b4_warmstart', {'head', 'head_params'}),
    ('c2_rank64_convpass192', 'a1_input', {'peft_kwargs', 'conv_adapters'}),
    ('b4_convpass_72ep', 'b4_convpass', {'epochs'}),
    ('b4_wst_72ep', 'b4_convpass_72ep', {'wst_stem'}),
    ('d1_96ep', 'b4_convpass_72ep', {'epochs'}),
    ('d2_ema', 'b4_convpass_72ep', {'ema_decay'}),
    ('d3_head512_cp352', 'b4_convpass_72ep', {'head_params', 'conv_adapters'}),
    ('d4_convhead_cp352', 'b4_convpass_72ep', {'head_params', 'conv_adapters'}),
    ('d5_depthwise640', 'b4_convpass_72ep', {'conv_adapters'}),
    ('d6_multiscale', 'b4_convpass_72ep', {'conv_adapters'}),
    ('d7_parallel', 'b4_convpass_72ep', {'conv_adapters'}),
    ('d8_rpn06', 'b4_convpass_72ep', {'rpn_fg_iou_thresh'}),
    ('d9_stem128', 'b4_convpass_72ep', {'detail_stem_params'}),
    ('d10_full_train', 'b4_convpass_72ep', {'val_fraction'}),
])
def test_each_ablation_changes_only_its_factor(name, parent, changed):
    parent = parent if isinstance(parent, Path) else CONFIGS / 'ablations' / f'{parent}.yaml'
    assert recipe_diff(load_config(CONFIGS / 'ablations' / f'{name}.yaml'), load_config(parent)) == changed


def test_run_names_are_unique():
    names = [load_config(path)['name'] for path in [BASE, *ABLATIONS]]
    assert len(names) == len(set(names))


def test_unknown_backbone_or_adapter_is_rejected(tmp_path):
    for text in ('backbone: {name: vit}', 'peft: {method: lora}'):
        path = tmp_path / 'bad.yaml'
        path.write_text(text)
        with pytest.raises(ValueError):
            load_config(path)


@pytest.mark.skipif(not os.environ.get('SARBENCH_TEST_DINOV3'),
                    reason='set SARBENCH_TEST_DINOV3=1; builds DINOv3 from its (gated) Hugging Face config')
@pytest.mark.parametrize('path', [BASE, *ABLATIONS], ids=lambda p: p.stem)
def test_every_config_fits_the_parameter_budget(path):
    from sarbench.model import MAX_TOTAL_PARAMETERS, MAX_TRAINABLE_PARAMETERS, build_model, parameter_counts
    args = SimpleNamespace(**{**load_config(path), 'pretrained': False})  # the count needs no weights
    counts = parameter_counts(*build_model(args))
    assert counts['total'] <= MAX_TOTAL_PARAMETERS and counts['trainable'] <= MAX_TRAINABLE_PARAMETERS
