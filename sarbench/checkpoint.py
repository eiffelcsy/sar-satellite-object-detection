"""Training checkpoints, so a run cut off by the cluster's time limit can continue in a new job.

A checkpoint holds what training changes: the trainable weights and every buffer, the optimizer and LR schedule
state, the RNG states, the last finished epoch, the validation history and the training time so far. The frozen
backbone weights are left out (~0.35 GB): `build_model` loads them from the same pretrained checkpoint every time,
and `load_checkpoint` checks that exactly those are the keys it did not restore.

The file is written to a temporary name and renamed, so a job killed mid-write leaves the previous checkpoint.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch


def _split_state_names(model):
    """state_dict names of the trainable tensors (parameters with requires_grad, and buffers) and of the frozen
    parameters. A module registered under two names (DINOv3's blocks: `dinov3.model.layer` and `_layers`) appears in
    the state_dict under both, so the names are read from the state_dict itself, not from named_parameters()."""
    buffer_ids = {id(b) for b in model.buffers()}
    trainable, frozen = set(), set()
    for name, value in model.state_dict(keep_vars=True).items():
        if id(value) in buffer_ids or value.requires_grad:
            trainable.add(name)
        else:
            frozen.add(name)
    return trainable, frozen


def trainable_state(model):
    """The model's trainable parameters and all its buffers, by state_dict name."""
    trainable, _ = _split_state_names(model)
    return {name: value for name, value in model.state_dict().items() if name in trainable}


def save_checkpoint(path, model, optimizer, schedule, epoch, history, elapsed, extra=None):
    path = Path(path)
    state = {'model': trainable_state(model), 'optimizer': optimizer.state_dict(), 'schedule': schedule.state_dict(),
             'epoch': epoch, 'history': history, 'elapsed': elapsed,
             'rng': {'torch': torch.get_rng_state(), 'python': random.getstate(), 'numpy': np.random.get_state(),
                     'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
             'extra': extra or {}}
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on one filesystem


def load_checkpoint(path, model, optimizer, schedule):
    """Restore a checkpoint into a freshly built model / optimizer / schedule; returns the checkpoint dict
    (epoch, history, elapsed, extra) without the tensors it restored."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state['model'], strict=False)
    _, frozen = _split_state_names(model)
    if unexpected or set(missing) != frozen:
        raise ValueError(f'checkpoint {path} does not match this model: unexpected {sorted(unexpected)[:5]}, '
                         f'missing trainable {sorted(set(missing) - frozen)[:5]}')
    optimizer.load_state_dict(state['optimizer'])
    schedule.load_state_dict(state['schedule'])
    rng = state['rng']
    torch.set_rng_state(rng['torch'])
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
    return {key: state[key] for key in ('epoch', 'history', 'elapsed', 'extra')}
