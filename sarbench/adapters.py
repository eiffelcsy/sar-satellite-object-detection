"""PEFT adapters (LoRA, MoE-LoRA) for the attention layers of a frozen backbone, behind a small registry.

Each backbone lists the attention projections to wrap via `adapter_targets()` (see backbones.py), so the same
adapter works for a fused-qkv ViT and a separate q/k/v/o ViT. The method and its hyper-parameters come from the
YAML configs (`peft.method` / `peft.params`).
"""
from torch import nn

ADAPTERS = {}


def register_adapter(name):
    """Register an adapter class under `name`, for add_adapters() and the YAML configs."""

    def decorator(cls):
        ADAPTERS[name] = cls
        return cls

    return decorator


@register_adapter('lora')
class LoRA(nn.Module):
    """y = W x + (alpha / r) * B A x, where W is the wrapped layer (frozen by add_adapters)."""

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)  # A
        self.up = nn.Linear(rank, base.out_features, bias=False)  # B
        nn.init.zeros_(self.up.weight)  # B = 0: training starts from exactly the pretrained layer
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.down(x))


@register_adapter('moelora')
class MoELoRA(nn.Module):
    """y = W x + (alpha / r) * sum_e g_e(x) B_e A_e x, with gates g(x) = softmax(router[task](x)) per token.

    The experts' A_e are stacked in `down` and their B_e side by side in `up`, so gating each
    expert's r hidden units and applying `up` sums the experts.
    """

    def __init__(self, base: nn.Linear, experts=4, rank=4, alpha=8, tasks=2):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, experts * rank, bias=False)  # A_1 ... A_E
        self.up = nn.Linear(experts * rank, base.out_features, bias=False)  # B_1 ... B_E
        nn.init.zeros_(self.up.weight)
        self.routers = nn.ModuleList(nn.Linear(base.in_features, experts, bias=False) for _ in range(tasks))
        self.rank = rank
        self.scale = alpha / rank
        self.task = 0  # which router to use; see set_task

    def forward(self, x):
        gates = self.routers[self.task](x).softmax(dim=-1)
        gates = gates.repeat_interleave(self.rank, dim=-1)  # expert e's gate on each of its r hidden units
        return self.base(x) + self.scale * self.up(self.down(x) * gates)


def add_adapters(backbone, kind, targets=None, **params):
    """Freeze the backbone, then wrap the projections selected by `backbone.adapter_targets(targets)`.

    kind: 'full' (no adapter, the backbone keeps training), 'lora' or 'moelora'. `targets` is the list of
    backbone-specific module groups to adapt (e.g. ['attn', 'mlp']; a new backbone documents its groups in
    `adapter_targets`). `params` are forwarded to the adapter constructor (rank, alpha, ...).
    """
    if kind == 'full':
        return
    if kind not in ADAPTERS:
        raise ValueError(f"unknown adapter '{kind}'; available: full, {sorted(ADAPTERS)}")
    adapter = ADAPTERS[kind]
    backbone.requires_grad_(False)
    pairs = backbone.adapter_targets(targets) if targets is not None else backbone.adapter_targets()
    for parent, name in pairs:
        setattr(parent, name, adapter(getattr(parent, name), **params))


def set_task(module, task):
    """Route every MoELoRA inside `module` through the router of `task` (0 = classification, 1 = detection)."""
    for layer in module.modules():
        if isinstance(layer, MoELoRA):
            layer.task = task
