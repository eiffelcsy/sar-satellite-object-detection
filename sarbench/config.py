"""Run configuration: a YAML file sets the DINOv3 checkpoint, the DoRA adaptation (with the per-block capacity
options) and may override the model, training, data, pre-processing and output options. train.py reads a config
with `--config` and lets explicit CLI flags override it.

Schema (every section is optional; the defaults are the `dinov3_dora_giou` recipe's backbone and adapter):

    base: ../dinov3_dora_giou.yaml # optional: start from this config (path relative to this file) and override
                                   # it section by section; nested mappings merge, everything else replaces
    name: dinov3_dora_giou         # run name (default: dinov3_dora)
    backbone:
      name: dinov3                 # the only backbone
      pretrained: true
      params: {model_name: facebook/dinov3-vitb16-pretrain-lvd1689m}
    peft:
      method: dora                 # the only adapter
      params: {rank: 32, alpha: 32}
      targets: [attn, mlp]         # q/k/v/o_proj and up/down_proj of every block
      rank_overrides: {0: 192}     # per-block DoRA rank (alpha / rank is kept)
      full_blocks: [0, 1]          # blocks trained entirely (no DoRA there), at train.unfrozen_lr
      conv_adapters: {layers: [0, 1, 2, 3], dim: 128, kernel: dense, placement: sequential}
                                   # Convpass adapters on these blocks; kernel dense | depthwise | multiscale,
                                   # placement sequential (after the block) | parallel (beside it)
    loss:                          # optional per-term weights on top of the summed loss (default 1.0 each)
      classification: 1.0
      loss_box_reg: 2.0            # Faster R-CNN RoI box loss (loss_rpn_box_reg for the RPN)
    model:
      detail_stem: true            # real stride-4 detail from the pixels, added to the neck's P2
      fusion_layers: [5, 8, 11]    # learnable weighted sum of these ViT blocks before the neck
      input_adapter: true          # ImageNet norm + learnable 1x1 conv (no-op init) + unfrozen patch embedding
      wst_stem: true               # wavelet scattering (J=2, L=8 -> 81 ch at stride 4) added to P2
      giou_weight: 1.0             # extra GIoU term added to the RoI box regression loss
      roi_sampling_ratio: 2        # RoIAlign sampling_ratio
      roi_output_size: 7           # RoIAlign output_size
      head: cascade                # standard (default) | cascade RoI head
      head_params: {num_stages: 3, share_head: false, hidden_dim: 256, fg_iou_thresholds: [0.4, 0.5, 0.6]}
                                   # both heads: box_head mlp (default) | conv (+ num_convs, conv_dim, reduce)
      detail_stem_params: {channels: 64, extra_convs: 0}   # P2 conv stem width / extra stride-4 3x3 convs
      rpn_fg_iou_thresh: 0.7       # RPN anchor positive / negative IoU thresholds
      rpn_bg_iou_thresh: 0.3
      score_thresh: 0.05           # inference post-processing only (sweep it with sweep_postprocess.py)
      nms_thresh: 0.5
      detections_per_img: 100
    train: {epochs: 48, stage_a_epochs: 0, stage_a_lr: 2.0e-4, batch_size: 16, lr: 1.0e-4,
            backbone_lr: null, unfrozen_lr: 2.0e-5, cls_lr: null, ema_decay: 0.0, warmup_epochs: 1,
            wd_exclude_1d: false, weight_decay: 0.05, eval_every: 4, workers: 8, seed: 0}
            # lr: new modules outside the backbone (neck, RPN, RoI head, stems, input adapter);
            # cls_lr: the image classifier (default lr); backbone_lr: new backbone parameters (DoRA, Convpass;
            # default lr);
            # unfrozen_lr: pretrained weights made trainable (patch embedding, full_blocks).
            # stage_a_epochs: first N epochs freeze the backbone and train the heads only at stage_a_lr.
            # warmup_epochs: linear learning-rate warmup before the cosine decay (stage B).
            # wd_exclude_1d: no weight decay on 1-D parameters (biases, norm weights, DoRA magnitudes, LayerScale).
            # ema_decay: > 0 keeps an exponential moving average of the trainable weights (e.g. 0.9998), used for
            # validation and saved as model.pt.
    data: {root: null, val_fraction: 0.1, limit: null, mosaic: 0.0, copy_paste: 0.0, copy_paste_margin: 0,
           scale_jitter: null, rotation90: false}
          # val_fraction: 0 trains on all of train (no held-out validation; for the final model)
          # split_seed (0): which images are held out; train.seed only changes the training randomness, so runs
          # with different seeds are scored on the same held-out images
    preprocess: {pseudo_rgb: false, mode: despeckle, edge: sobel, despeckle: true, clip_percentile: 0.5,
                 cache: null, bm3d_sigma: null, bm3d_profile: np, bm3d_threads: 1}
                 # pseudo_rgb: 1-channel SAR -> 3 channels; mode: despeckle (amp+base+edge),
                 # amp_log_edge (amp+log-amp+edge) or amplitude (amp repeated); edge: sobel | highpass.
    output: {dir: runs, wandb: false, wandb_project: sar-satellite-object-detection, wandb_run_name: null}
"""
from pathlib import Path

# Defaults match the argparse defaults in train.py, so an absent config (plain CLI) behaves the same.
DEFAULTS = {
    'pretrained': True,
    'backbone_kwargs': {'model_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m'},
    'peft_kwargs': {'rank': 32, 'alpha': 32},
    'peft_targets': ['attn', 'mlp'],
    'rank_overrides': {},
    'full_blocks': [],
    'conv_adapters': {},
    'loss_weights': {},
    'detail_stem': False,
    'fusion_layers': None,
    'input_adapter': False,
    'wst_stem': False,
    'wst_params': {},
    'giou_weight': 0.0,
    'roi_sampling_ratio': 2,
    'roi_output_size': 7,
    'head': 'standard',
    'head_params': {},
    'detail_stem_params': {},
    'rpn_fg_iou_thresh': 0.7,
    'rpn_bg_iou_thresh': 0.3,
    'score_thresh': 0.05,
    'nms_thresh': 0.5,
    'detections_per_img': 100,
    'name': None,
    'epochs': 24,
    'stage_a_epochs': 0,
    'stage_a_lr': 2e-4,
    'batch_size': 16,
    'lr': 1e-4,
    'backbone_lr': None,
    'unfrozen_lr': 2e-5,
    'cls_lr': None,
    'ema_decay': 0.0,
    'warmup_epochs': 1,
    'wd_exclude_1d': False,
    'weight_decay': 0.05,
    'eval_every': 4,
    'workers': 8,
    'seed': 0,
    'data': None,
    'val_fraction': 0.1,
    'split_seed': 0,
    'limit': None,
    'mosaic': 0.0,
    'copy_paste': 0.0,
    'copy_paste_margin': 0,
    'scale_jitter': None,
    'rotation90': False,
    'out': 'runs',
    'pseudo_rgb': False,
    'pseudo_rgb_mode': 'despeckle',
    'edge': 'sobel',
    'no_despeckle': False,
    'clip_percentile': 0.5,
    'preprocess_cache': None,
    'bm3d_sigma': None,
    'bm3d_profile': 'np',
    'bm3d_threads': 1,
    'wandb': False,
    'wandb_project': 'sar-satellite-object-detection',
    'wandb_run_name': None,
}

_MODEL_KEYS = ('detail_stem', 'fusion_layers', 'input_adapter', 'wst_stem', 'wst_params', 'giou_weight',
               'roi_sampling_ratio', 'roi_output_size', 'head', 'head_params', 'detail_stem_params',
               'rpn_fg_iou_thresh', 'rpn_bg_iou_thresh', 'score_thresh', 'nms_thresh', 'detections_per_img')
_TRAIN_KEYS = ('epochs', 'stage_a_epochs', 'stage_a_lr', 'batch_size', 'lr', 'backbone_lr', 'unfrozen_lr',
               'cls_lr', 'ema_decay', 'warmup_epochs', 'wd_exclude_1d', 'weight_decay', 'eval_every', 'workers',
               'seed')
_DATA_KEYS = {'root': 'data', 'val_fraction': 'val_fraction', 'split_seed': 'split_seed', 'limit': 'limit',
              'mosaic': 'mosaic', 'copy_paste': 'copy_paste', 'copy_paste_margin': 'copy_paste_margin',
              'scale_jitter': 'scale_jitter', 'rotation90': 'rotation90'}
_PREPROCESS_KEYS = {'pseudo_rgb': 'pseudo_rgb', 'mode': 'pseudo_rgb_mode', 'edge': 'edge',
                    'clip_percentile': 'clip_percentile', 'cache': 'preprocess_cache', 'bm3d_sigma': 'bm3d_sigma',
                    'bm3d_profile': 'bm3d_profile', 'bm3d_threads': 'bm3d_threads'}
_OUTPUT_KEYS = {'dir': 'out', 'wandb': 'wandb', 'wandb_project': 'wandb_project', 'wandb_run_name': 'wandb_run_name'}


def _merge(base, override):
    """Recursively merge `override` into `base`: nested mappings merge, any other value replaces."""
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) \
            else value
    return merged


def read_raw(path):
    """The YAML mapping of `path`, with its `base:` chain resolved (each base relative to the file naming it)."""
    import yaml  # only needed when --config is used

    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if raw.get('base'):
        raw = _merge(read_raw(path.parent / raw.pop('base')), raw)
    return raw


def load_config(path):
    """Read a YAML run config and flatten it into the argparse attribute names used by train.py."""
    raw = read_raw(path)
    config = {key: (value.copy() if isinstance(value, (dict, list)) else value) for key, value in DEFAULTS.items()}

    backbone = raw.get('backbone') or {}
    if backbone.get('name', 'dinov3') != 'dinov3':
        raise ValueError(f"{path}: only the dinov3 backbone is supported, got {backbone['name']!r}")
    config['pretrained'] = bool(backbone.get('pretrained', True))
    config['backbone_kwargs'].update(backbone.get('params') or {})

    peft = raw.get('peft') or {}
    if peft.get('method', 'dora') != 'dora':
        raise ValueError(f"{path}: only the dora adapter is supported, got {peft['method']!r}")
    config['peft_kwargs'].update(peft.get('params') or {})
    if peft.get('targets'):
        config['peft_targets'] = list(peft['targets'])
    config['rank_overrides'] = {int(k): int(v) for k, v in (peft.get('rank_overrides') or {}).items()}
    config['full_blocks'] = [int(i) for i in peft.get('full_blocks') or []]
    config['conv_adapters'] = dict(peft.get('conv_adapters') or {})

    config['loss_weights'] = dict(raw.get('loss') or {})
    model = raw.get('model') or {}
    for key in _MODEL_KEYS:
        if key in model:
            config[key] = model[key]
    if config['fusion_layers'] is not None:
        config['fusion_layers'] = list(config['fusion_layers']) or None

    if raw.get('name') is not None:
        config['name'] = raw['name']

    train = raw.get('train') or {}
    for key in _TRAIN_KEYS:
        if key in train:
            config[key] = train[key]
    for section, keys in (('data', _DATA_KEYS), ('preprocess', _PREPROCESS_KEYS), ('output', _OUTPUT_KEYS)):
        section = raw.get(section) or {}
        for key, dest in keys.items():
            if key in section:
                config[dest] = section[key]
    preprocess = raw.get('preprocess') or {}
    if 'despeckle' in preprocess:
        config['no_despeckle'] = not preprocess['despeckle']
    return config
