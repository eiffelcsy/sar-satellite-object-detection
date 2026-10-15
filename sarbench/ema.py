"""Exponential moving average (EMA) of a model's trainable weights: one model, averaged over its own training steps.

shadow <- d * shadow + (1 - d) * weight after every optimizer step, with the decay ramped up from ~0.1 early on
(d = min(decay, (1 + n) / (10 + n)) after n updates) so the average is not anchored to the random head init.
Only trainable parameters are tracked (the frozen backbone never changes), and the model has no BatchNorm
running statistics to average.
"""
from contextlib import contextmanager

import torch


class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.updates = 0
        self.pairs = [(p, p.detach().clone()) for p in model.parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for weight, shadow in self.pairs:
            shadow.lerp_(weight.detach(), 1 - decay)  # shadow + (1 - decay) * (weight - shadow)

    @torch.no_grad()
    def copy_to_model(self):
        """Overwrite the model's weights with the averages (for the final evaluation and model.pt)."""
        for weight, shadow in self.pairs:
            weight.copy_(shadow)

    @contextmanager
    def averaged(self):
        """Temporarily swap the averaged weights in (for validation), restoring the training weights after."""
        backup = [weight.detach().clone() for weight, _ in self.pairs]
        self.copy_to_model()
        try:
            yield
        finally:
            with torch.no_grad():
                for (weight, _), saved in zip(self.pairs, backup):
                    weight.copy_(saved)
