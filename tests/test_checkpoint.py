import pytest
import torch
from torch import nn

from sarbench.checkpoint import load_checkpoint, save_checkpoint
from sarbench.ema import ModelEMA


def _setup(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    model[0].weight.requires_grad_(False)  # a frozen "pretrained" weight, rebuilt identically by the seed
    model.register_buffer('scale', torch.ones(2))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step))
    return model, optimizer, schedule, ModelEMA(model, 0.9)


def _steps(model, optimizer, schedule, ema, n):
    for _ in range(n):
        loss = model(torch.randn(16, 4)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        ema.update()


def test_a_resumed_run_continues_exactly_like_an_uninterrupted_one(tmp_path):
    straight = _setup()
    _steps(*straight, 6)

    first = _setup()
    _steps(*first, 3)
    save_checkpoint(tmp_path / 'last.pt', *first[:3], epoch=3, history=[{'epoch': 3}], elapsed=12.5, ema=first[3],
                    extra={'wandb_id': 'abc'})
    torch.randn(100)  # the RNG moves on before the "job" dies; the checkpoint restores it
    resumed = _setup()
    state = load_checkpoint(tmp_path / 'last.pt', *resumed[:3], ema=resumed[3])
    assert state == {'epoch': 3, 'history': [{'epoch': 3}], 'elapsed': 12.5, 'extra': {'wandb_id': 'abc'}}
    _steps(*resumed, 3)

    for a, b in zip(straight[0].state_dict().values(), resumed[0].state_dict().values()):
        assert torch.equal(a, b)
    assert resumed[2].last_epoch == straight[2].last_epoch == 6
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(straight[3].pairs, resumed[3].pairs))
    assert not (tmp_path / 'last.pt.tmp').exists()


def test_the_checkpoint_skips_frozen_weights_and_rejects_another_model(tmp_path):
    model, optimizer, schedule, _ = _setup()
    save_checkpoint(tmp_path / 'last.pt', model, optimizer, schedule, epoch=1, history=[], elapsed=0.0)
    saved = torch.load(tmp_path / 'last.pt', weights_only=False)['model']
    assert '0.weight' not in saved and {'0.bias', '2.weight', '2.bias', 'scale'} == set(saved)
    other = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    other_opt = torch.optim.AdamW(other.parameters())
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(tmp_path / 'last.pt', other, other_opt, torch.optim.lr_scheduler.LambdaLR(other_opt, lambda s: 1))
