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
    ('d10_full_train', 'e6_stem_200ep', {'val_fraction'}),
    ('e1_stem_96ep', 'd1_96ep', {'detail_stem_params'}),
    ('e2_120ep', 'd1_96ep', {'epochs'}),
    ('e3_d1_seed1', 'd1_96ep', {'seed'}),
    ('e4_stem_120ep', 'e1_stem_96ep', {'epochs'}),
    ('e5_stem_150ep', 'e4_stem_120ep', {'epochs'}),
    ('e6_stem_200ep', 'e5_stem_150ep', {'epochs'}),
    ('f1_adapter_lr3e-4', 'e1_stem_96ep', {'backbone_lr'}),
    ('f2_adapter_lr1e-3', 'e1_stem_96ep', {'backbone_lr', 'warmup_epochs'}),
    ('f3_lr2e-4', 'e1_stem_96ep', {'lr'}),
    ('f4_wd_exclude_1d', 'e1_stem_96ep', {'wd_exclude_1d'}),
    ('f5_split_lr', 'e1_stem_96ep', {'backbone_lr', 'lr', 'cls_lr'}),
    ('g1_split_lr_200ep', 'e6_stem_200ep', {'backbone_lr', 'lr', 'cls_lr'}),
    ('d11_full_train_g1_250ep', 'g1_split_lr_200ep', {'epochs', 'val_fraction'}),
    ('d12_full_train_g1', 'g1_split_lr_200ep', {'val_fraction'}),
    ('h1_vit_adapter', 'f5_split_lr', {'detail_stem', 'vit_adapter'}),
    ('h2_patch8', 'f5_split_lr', {'backbone_kwargs'}),
    ('h3_nwd', 'f5_split_lr', {'box_similarity'}),
    ('h4_iou_aware', 'f5_split_lr', {'iou_aware'}),
    ('h5_lora_plus', 'f5_split_lr', {'lora_plus_ratio'}),
    ('h6_p2_heatmap', 'f5_split_lr', {'heatmap_aux'}),
    ('i1_cls_loss025_200ep', 'g1_split_lr_200ep', {'loss_weights'}),
    ('i2_cls_detach_200ep', 'g1_split_lr_200ep', {'cls_detach'}),
    ('d13_full_train_i2_250ep', 'i2_cls_detach_200ep', {'epochs', 'val_fraction'}),
    ('d14_full_train_i1_cls01_250ep', 'i1_cls_loss025_200ep', {'loss_weights', 'epochs', 'val_fraction'}),
    ('j0_base', 'f5_split_lr', {'loss_weights', 'mosaic', 'copy_paste_same_class', 'copy_paste_margin'}),
    ('j1_roi_fused', 'j0_base', {'roi_pooling'}),
    ('j2_roi_canonical56', 'j0_base', {'roi_canonical_scale'}),
    ('j3_cls_token', 'j0_base', {'cls_input'}),
    ('d15_full_train_j0_250ep', 'd14_full_train_i1_cls01_250ep', {'mosaic', 'copy_paste_same_class', 'copy_paste_margin'}),
    ('d16_full_train_j1_250ep', 'd15_full_train_j0_250ep', {'roi_pooling'}),
    ('d17_full_train_j1_200ep', 'd16_full_train_j1_250ep', {'epochs'}),
    ('k1_copy_paste_small', 'j1_roi_fused', {'copy_paste_small'}),
    ('k2_roi_sparse', 'j1_roi_fused', {'roi_batch_size_per_image', 'roi_positive_fraction'}),
    ('k3_iou_aware', 'j1_roi_fused', {'iou_aware'}),
    ('k4_giou_weight2', 'j1_roi_fused', {'giou_weight'}),
    ('k5_box_loss4', 'j1_roi_fused', {'loss_weights'}),
    ('k6_no_copy_paste', 'j1_roi_fused', {'copy_paste'}),
    ('k7_alpha_giou', 'j1_roi_fused', {'giou_alpha'}),
    ('k8_patch8_bs32', 'j1_roi_fused', {'backbone_kwargs', 'batch_size', 'lr', 'backbone_lr', 'cls_lr', 'unfrozen_lr'}),
    ('k9_p2_heatmap', 'j1_roi_fused', {'heatmap_aux'}),
    ('k10_scale_jitter', 'j1_roi_fused', {'scale_jitter'}),
    ('k11_roi_aligned', 'j1_roi_fused', {'roi_aligned'}),
    ('k12_roi_stem_aligned', 'k11_roi_aligned', {'detail_stem_params'}),
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


def test_training_seed_does_not_move_the_held_out_split():
    """e3 changes only the training seed; its held-out images are d1's, so the two are scored on the same split."""
    d1, e3 = (load_config(CONFIGS / 'ablations' / f'{name}.yaml') for name in ('d1_96ep', 'e3_d1_seed1'))
    assert (d1['seed'], e3['seed']) == (0, 1) and d1['split_seed'] == e3['split_seed'] == 0


def test_a_bare_yaml_off_turns_rescoring_off(tmp_path):
    path = tmp_path / 'rescore.yaml'
    path.write_text('model: {rescore: off}')  # YAML 1.1 reads it as false
    assert load_config(path)['rescore'] == 'off'
