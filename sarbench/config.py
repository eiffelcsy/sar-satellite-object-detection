"""Run configuration: a YAML file names the backbone and the PEFT method (with their parameters) and may
override the training, data, pre-processing and output options. train.py reads a config with `--config` and
lets explicit CLI flags override it, so the old command line keeps working.

Schema (only `backbone` and `peft` are required):

    name: vit_lora                 # optional run name (default: <backbone>_<init>_<adapt>)
    backbone:
      name: vit
      pretrained: true             # false -> random weights ('scratch')
      params: {}                   # forwarded to the backbone constructor
    peft:
      method: dora                 # full | lora | dora | moelora
      params: {rank: 16, alpha: 32}  # forwarded to the adapter constructor
      targets: [attn, mlp]         # optional; backbone module groups to adapt (default: the backbone's own)
    loss:                          # optional per-term weights on top of the summed loss (default 1.0 each)
      classification: 1.0
      loss_box_reg: 2.0            # Faster R-CNN RoI box loss (loss_rpn_box_reg for the RPN)
    model:
      detail_stem: true            # real stride-4 detail from the pixels, added to the neck's P2
      fusion_layers: [5, 8, 11]    # optional: learnable weighted sum of these ViT blocks before the neck
      input_adapter: true          # ImageNet norm + learnable 1x1 conv (no-op init) + unfrozen patch embedding
      wst_stem: true               # wavelet scattering (J=2, L=8 -> 81 ch at stride 4) added to P2
      head: cascade                # standard (default) | deform | cascade RoI head
      head_params: {num_stages: 3} # forwarded to the head (deform: kernel_size/num_convs; cascade: num_stages)
      focal_loss: true             # focal loss for the RPN objectness, RoI classifier and image classifier
      focal_gamma: 2.0
      focal_alpha: 0.25            # RPN (binary) focal only
      giou_weight: 1.0             # extra GIoU term added to the RoI box regression loss (all heads)
      roi_sampling_ratio: 4        # RoIAlign sampling_ratio (applies to every head)
      roi_output_size: 7           # RoIAlign output_size
    train: {epochs: 48, stage_a_epochs: 0, stage_a_lr: 2.0e-4, batch_size: 16, lr: 1.0e-4,
            backbone_lr: null, weight_decay: 0.05, eval_every: 4, workers: 8, seed: 0}
            # stage_a_epochs: first N epochs freeze the backbone (incl. DoRA) and train heads only at
            # stage_a_lr; the remaining `epochs` then run the normal recipe with DoRA switched on.
    data: {root: null, val_fraction: 0.1, limit: null, mosaic: 0.1, copy_paste: 0.5,
           scale_jitter: [0.8, 1.25], rotation90: true, preprocess_after_aug: false}
          # mosaic / copy_paste: train-time augmentation probabilities (0 disables each)
          # scale_jitter / rotation90: train-time object-scale protection (rotation90 = 0/90/180/270)
          # preprocess_after_aug: false (default) filters before augmentation and caches; true filters the
          # augmented pixels (geometrically exact, but BM3D reruns every sample and cannot be cached)
    preprocess: {enabled: false, pseudo_rgb: false, edge: sobel, despeckle: true, log: true,
                 clip_percentile: 0.5, cache: null, bm3d_sigma: null, bm3d_profile: np, bm3d_threads: 1,
                 mode: despeckle}
                 # pseudo_rgb: DINOv3 input -> 3 channels; mode: despeckle (amp+base+edge),
                 # amp_log_edge (amp+log-amp+edge) or amplitude (amp repeated); edge: sobel | highpass.
                 # Mutually exclusive with `enabled` (plain SARPreprocess). Filter channels are computed
                 # after the geometric augmentations whenever pre-processing and training are both on.
    output: {dir: runs, wandb: false, wandb_project: sar-satellite-object-detection, wandb_run_name: null}
"""
from pathlib import Path

# Defaults match the argparse defaults in train.py, so an absent config (plain CLI) behaves as before.
DEFAULTS = {
    'backbone': 'vit',
    'init': 'pretrained',
    'adapt': 'full',
    'backbone_kwargs': {},
    'peft_kwargs': {},
    'peft_targets': None,
    'loss_weights': {},
    'detail_stem': False,
    'fusion_layers': None,
    'input_adapter': False,
    'wst_stem': False,
    'wst_params': {},
    'head': 'standard',
    'head_params': {},
    'focal_loss': False,
    'focal_gamma': 2.0,
    'focal_alpha': 0.25,
    'giou_weight': 0.0,
    'roi_sampling_ratio': 2,
    'roi_output_size': 7,
    'name': None,
    'epochs': 24,
    'stage_a_epochs': 0,
    'stage_a_lr': 2e-4,
    'batch_size': 16,
    'lr': 1e-4,
    'backbone_lr': None,
    'weight_decay': 0.05,
    'eval_every': 4,
    'workers': 8,
    'seed': 0,
    'data': None,
    'val_fraction': 0.1,
    'limit': None,
    'mosaic': 0.0,
    'copy_paste': 0.0,
    'scale_jitter': None,
    'rotation90': False,
    'preprocess_after_aug': False,
    'out': 'runs',
    'preprocess': False,
    'pseudo_rgb': False,
    'pseudo_rgb_mode': 'despeckle',
    'edge': 'sobel',
    'no_despeckle': False,
    'no_log_transform': False,
    'clip_percentile': 0.5,
    'preprocess_cache': None,
    'bm3d_sigma': None,
    'bm3d_profile': 'np',
    'bm3d_threads': 1,
    'wandb': False,
    'wandb_project': 'sar-satellite-object-detection',
    'wandb_run_name': None,
}

_TRAIN_KEYS = ('epochs', 'stage_a_epochs', 'stage_a_lr', 'batch_size', 'lr', 'backbone_lr', 'weight_decay',
               'eval_every', 'workers', 'seed')
_DATA_KEYS = {'root': 'data', 'val_fraction': 'val_fraction', 'limit': 'limit',
              'mosaic': 'mosaic', 'copy_paste': 'copy_paste',
              'scale_jitter': 'scale_jitter', 'rotation90': 'rotation90',
              'preprocess_after_aug': 'preprocess_after_aug'}
_OUTPUT_KEYS = {'dir': 'out', 'wandb': 'wandb', 'wandb_project': 'wandb_project', 'wandb_run_name': 'wandb_run_name'}


def load_config(path):
    """Read a YAML run config and flatten it into the argparse attribute names used by train.py."""
    import yaml  # only needed when --config is used

    raw = yaml.safe_load(Path(path).read_text()) or {}
    config = dict(DEFAULTS)

    backbone = raw.get('backbone') or {}
    if isinstance(backbone, str):
        backbone = {'name': backbone}
    config['backbone'] = backbone.get('name', config['backbone'])
    config['init'] = 'pretrained' if backbone.get('pretrained', True) else 'scratch'
    config['backbone_kwargs'] = dict(backbone.get('params') or {})

    peft = raw.get('peft') or {}
    if isinstance(peft, str):
        peft = {'method': peft}
    config['adapt'] = peft.get('method', config['adapt'])
    config['peft_kwargs'] = dict(peft.get('params') or {})
    config['peft_targets'] = list(peft['targets']) if peft.get('targets') else None

    config['loss_weights'] = dict(raw.get('loss') or {})
    model = raw.get('model') or {}
    if 'detail_stem' in model:
        config['detail_stem'] = bool(model['detail_stem'])
    if 'fusion_layers' in model:
        config['fusion_layers'] = list(model['fusion_layers']) or None
    if 'input_adapter' in model:
        config['input_adapter'] = bool(model['input_adapter'])
    if 'wst_stem' in model:
        config['wst_stem'] = bool(model['wst_stem'])
    if 'wst_params' in model:
        config['wst_params'] = dict(model['wst_params'] or {})
    if 'head' in model:
        config['head'] = model['head']
    if 'head_params' in model:
        config['head_params'] = dict(model['head_params'] or {})
    if 'focal_loss' in model:
        config['focal_loss'] = bool(model['focal_loss'])
    for key in ('focal_gamma', 'focal_alpha', 'giou_weight', 'roi_sampling_ratio', 'roi_output_size'):
        if key in model:
            config[key] = model[key]

    if raw.get('name') is not None:
        config['name'] = raw['name']

    train = raw.get('train') or {}
    for key in _TRAIN_KEYS:
        if key in train:
            config[key] = train[key]
    for section, keys in (('data', _DATA_KEYS), ('output', _OUTPUT_KEYS)):
        section = raw.get(section) or {}
        for key, dest in keys.items():
            if key in section:
                config[dest] = section[key]

    preprocess = raw.get('preprocess') or {}
    if 'enabled' in preprocess:
        config['preprocess'] = bool(preprocess['enabled'])
    if 'pseudo_rgb' in preprocess:
        config['pseudo_rgb'] = bool(preprocess['pseudo_rgb'])
    if 'edge' in preprocess:
        config['edge'] = preprocess['edge']
    if 'mode' in preprocess:
        config['pseudo_rgb_mode'] = preprocess['mode']
    if 'despeckle' in preprocess:
        config['no_despeckle'] = not preprocess['despeckle']
    if 'log' in preprocess:
        config['no_log_transform'] = not preprocess['log']
    for key in ('clip_percentile', 'cache', 'bm3d_sigma', 'bm3d_profile', 'bm3d_threads'):
        if key in preprocess:
            config['preprocess_cache' if key == 'cache' else key] = preprocess[key]

    return config
