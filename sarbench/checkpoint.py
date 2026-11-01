"""Training checkpoints, so a run cut off by the cluster's time limit can continue in a new job.

A checkpoint holds what training changes: the trainable weights and every buffer, the optimizer and LR schedule
state, the EMA shadow weights, the RNG states, the last finished epoch, the validation history and the training
time so far. The frozen backbone weights are left out (~0.35 GB): `build_model` loads them from the same pretrained
checkpoint every time, and `load_checkpoint` checks that exactly those are the keys it did not restore.

The file is written to a temporary name and renamed, so a job killed mid-write leaves the previous checkpoint.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch


def trainable_state(model):
    """The model's trainable parameters and all its buffers, by state_dict name."""
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    buffers = {name for name, _ in model.named_buffers()}
    return {name: value for name, value in model.state_dict().items() if name in trainable | buffers}


def save_checkpoint(path, model, optimizer, schedule, epoch, history, elapsed, ema=None, extra=None):
    path = Path(path)
    state = {'model': trainable_state(model), 'optimizer': optimizer.state_dict(), 'schedule': schedule.state_dict(),
             'epoch': epoch, 'history': history, 'elapsed': elapsed,
             'ema': None if ema is None else {'updates': ema.updates, 'shadow': [s for _, s in ema.pairs]},
             'rng': {'torch': torch.get_rng_state(), 'python': random.getstate(), 'numpy': np.random.get_state(),
                     'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
             'extra': extra or {}}
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic on one filesystem


def load_checkpoint(path, model, optimizer, schedule, ema=None):
    """Restore a checkpoint into a freshly built model / optimizer / schedule (/ EMA); returns the checkpoint dict
    (epoch, history, elapsed, extra) without the tensors it restored."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state['model'], strict=False)
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    if unexpected or set(missing) != frozen:
        raise ValueError(f'checkpoint {path} does not match this model: unexpected {sorted(unexpected)[:5]}, '
                         f'missing trainable {sorted(set(missing) - frozen)[:5]}')
    optimizer.load_state_dict(state['optimizer'])
    schedule.load_state_dict(state['schedule'])
    if ema is not None:
        if state['ema'] is None:
            raise ValueError(f'checkpoint {path} has no EMA weights, but ema_decay > 0')
        ema.updates = state['ema']['updates']
        with torch.no_grad():
            for (_, shadow), saved in zip(ema.pairs, state['ema']['shadow']):
                shadow.copy_(saved)
    rng = state['rng']
    torch.set_rng_state(rng['torch'])
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
    return {key: state[key] for key in ('epoch', 'history', 'elapsed', 'extra')}
