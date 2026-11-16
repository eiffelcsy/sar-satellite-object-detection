"""Run configuration: one YAML file (configs/final.yaml) holds every setting of the final recipe.

The file must list exactly the keys of `REQUIRED` (no defaults hide in the code), so the config is the single
source of truth for training and inference.
"""
from pathlib import Path

import yaml

REQUIRED = {
    'name': None,
    'backbone': ('model_name',),
    'adapters': ('dora_rank', 'dora_alpha', 'dora_targets', 'convpass_dim'),
    'model': ('stem_channels', 'stem_extra_convs', 'giou_weight', 'score_thresh', 'nms_thresh',
              'detections_per_img'),
    'preprocess': ('clip_percentile', 'cache'),
    'data': ('val_fraction', 'split_seed', 'copy_paste', 'copy_paste_margin'),
    'loss': ('classification', 'loss_box_reg', 'loss_rpn_box_reg'),
    'train': ('epochs', 'batch_size', 'lr', 'backbone_lr', 'cls_lr', 'unfrozen_lr', 'weight_decay', 'warmup_epochs',
              'eval_every', 'checkpoint_every', 'workers', 'seed'),
}


def load_config(path):
    """The YAML mapping at `path`, checked against `REQUIRED` (missing or unknown keys raise ValueError)."""
    config = yaml.safe_load(Path(path).read_text()) or {}
    if set(config) != set(REQUIRED):
        raise ValueError(f'{path}: sections must be {sorted(REQUIRED)}, got {sorted(config)}')
    for section, keys in REQUIRED.items():
        if keys is None:
            continue
        got = set(config[section] or {})
        if got != set(keys):
            raise ValueError(f'{path}: [{section}] missing {sorted(set(keys) - got)}, '
                             f'unknown {sorted(got - set(keys))}')
    return config
