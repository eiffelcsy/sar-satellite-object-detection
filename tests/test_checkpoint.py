import pytest
import torch
from torch import nn

from sarbench.checkpoint import load_checkpoint, save_checkpoint


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        self.alias = self.layers[2]  # one module under two names, as DINOv3's blocks (dinov3.model.layer, _layers)
        self.register_buffer('scale', torch.ones(2))

    def forward(self, x):
        return self.layers(x) * self.scale


def _setup():
    torch.manual_seed(0)
    model = _Net()
    model.layers[0].weight.requires_grad_(False)  # a frozen "pretrained" weight, rebuilt identically by the seed
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    return model, optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step))


def _steps(model, optimizer, schedule, n):
    for _ in range(n):
        loss = model(torch.randn(16, 4)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()


def test_a_resumed_run_continues_exactly_like_an_uninterrupted_one(tmp_path):
    straight = _setup()
    _steps(*straight, 6)
    first = _setup()
    _steps(*first, 3)
    save_checkpoint(tmp_path / 'last.pt', *first, epoch=3, history=[{'epoch': 3}], elapsed=12.5,
                    extra={'wandb_id': 'abc'})
    torch.randn(100)  # the RNG moves on before the "job" dies; the checkpoint restores it
    resumed = _setup()
    state = load_checkpoint(tmp_path / 'last.pt', *resumed)
    assert state == {'epoch': 3, 'history': [{'epoch': 3}], 'elapsed': 12.5, 'extra': {'wandb_id': 'abc'}}
    _steps(*resumed, 3)
    assert all(torch.equal(a, b) for a, b in zip(straight[0].state_dict().values(), resumed[0].state_dict().values()))


def test_the_checkpoint_skips_frozen_weights_and_rejects_another_model(tmp_path):
    save_checkpoint(tmp_path / 'last.pt', *_setup(), epoch=1, history=[], elapsed=0.0)
    assert set(torch.load(tmp_path / 'last.pt', weights_only=False)['model']) == {
        'layers.0.bias', 'layers.2.weight', 'layers.2.bias', 'alias.weight', 'alias.bias', 'scale'}
    other = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = torch.optim.AdamW(other.parameters())
    with pytest.raises((ValueError, RuntimeError)):
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1)
        load_checkpoint(tmp_path / 'last.pt', other, optimizer, schedule)
