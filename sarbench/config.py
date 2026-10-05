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
      method: lora                 # full | lora | moelora
      params: {rank: 16, alpha: 32}  # forwarded to the adapter constructor
      targets: [attn, mlp]         # optional; backbone module groups to adapt (default: the backbone's own)
    train: {epochs: 24, batch_size: 16, lr: 1.0e-4, backbone_lr: null, weight_decay: 0.05,
            eval_every: 4, workers: 8, seed: 0}
    data: {root: null, val_fraction: 0.1, limit: null}
    preprocess: {enabled: false, pseudo_rgb: false, edge: sobel, despeckle: true, log: true,
                 clip_percentile: 0.5, cache: null, bm3d_sigma: null, bm3d_profile: np, bm3d_threads: 1}
                 # pseudo_rgb: DINOv3 input (amplitude + despeckled base + Sobel/high-pass edges) -> 3 channels;
                 # edge: sobel | highpass. Mutually exclusive with `enabled` (plain SARPreprocess).
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
    'name': None,
    'epochs': 24,
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
    'out': 'runs',
    'preprocess': False,
    'pseudo_rgb': False,
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

_TRAIN_KEYS = ('epochs', 'batch_size', 'lr', 'backbone_lr', 'weight_decay', 'eval_every', 'workers', 'seed')
_DATA_KEYS = {'root': 'data', 'val_fraction': 'val_fraction', 'limit': 'limit'}
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
    if 'despeckle' in preprocess:
        config['no_despeckle'] = not preprocess['despeckle']
    if 'log' in preprocess:
        config['no_log_transform'] = not preprocess['log']
    for key in ('clip_percentile', 'cache', 'bm3d_sigma', 'bm3d_profile', 'bm3d_threads'):
        if key in preprocess:
            config['preprocess_cache' if key == 'cache' else key] = preprocess[key]

    return config
