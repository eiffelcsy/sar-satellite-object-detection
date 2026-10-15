"""The exponential moving average of the trainable weights (CPU only)."""
import torch
from torch import nn

from sarbench.ema import ModelEMA


def _model():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    model[0].requires_grad_(False)  # a frozen part is not tracked
    return model


def test_ema_tracks_only_trainable_weights_and_follows_the_average():
    model = _model()
    ema = ModelEMA(model, decay=0.5)
    assert len(ema.pairs) == 2  # the second layer's weight and bias
    start = model[1].weight.detach().clone()
    with torch.no_grad():
        model[1].weight.add_(1.0)
    ema.update()  # first update: decay = min(0.5, 2 / 11)
    torch.testing.assert_close(ema.pairs[0][1], start + (1 - 2 / 11) * 1.0)


def test_averaged_swaps_in_and_restores_the_training_weights():
    model = _model()
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model[1].weight.add_(1.0)
    trained = model[1].weight.detach().clone()
    with ema.averaged():
        assert not torch.equal(model[1].weight, trained)  # the (older) average is in place
    torch.testing.assert_close(model[1].weight, trained)
    ema.copy_to_model()
    assert not torch.equal(model[1].weight, trained)
