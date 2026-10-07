"""PEFT adapters (LoRA, DoRA, MoE-LoRA, MoE-DoRA) for the attention layers of a frozen backbone, behind a registry.

Each backbone lists the attention projections to wrap via `adapter_targets()` (see backbones.py), so the same
adapter works for a fused-qkv ViT and a separate q/k/v/o ViT. The method and its hyper-parameters come from the
YAML configs (`peft.method` / `peft.params`).
"""
import torch
import torch.nn.functional as F
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


@register_adapter('dora')
class DoRA(nn.Module):
    """Weight-decomposed LoRA (Liu et al., 2024): W' = m * (W + (alpha / r) B A) / ||W + (alpha / r) B A||_c.

    `m` is a per-output-neuron magnitude initialized to the frozen weight's column norm, and the update is
    renormalized, so init is an exact no-op and training starts from the pretrained layer (as LoRA does).
    """

    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)  # A
        self.up = nn.Linear(rank, base.out_features, bias=False)  # B
        nn.init.zeros_(self.up.weight)  # B = 0 -> W' = m * W / ||W||_c = W at init
        self.scale = alpha / rank
        self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1))  # ||W||_c per output neuron

    def forward(self, x):
        weight = self.base.weight + self.scale * (self.up.weight @ self.down.weight)
        weight = self.magnitude.unsqueeze(1) * weight / (weight.norm(dim=1, keepdim=True) + 1e-6)
        return F.linear(x, weight, self.base.bias)


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


@register_adapter('moedora')
class MoEDoRA(nn.Module):
    """Mixture-of-experts DoRA: each expert is a DoRA update, mixed per token by a softmax router.

    Expert e forms `W_e = W + (alpha / r) B_e A_e`, is weight-decomposed and renormalized
    (`m_e * W_e / ||W_e||_c`, with a learned per-output magnitude `m_e`), and the task router gates the experts:
    `y = sum_e g_e(x) * x W_e'^T + b`. At init `B_e = 0` and `m_e = ||W||_c`, so `W_e' = W` and the layer is an
    exact no-op. (DoRA renormalizes a single weight matrix, so here it is applied per expert before gating.)
    """

    def __init__(self, base: nn.Linear, experts=4, rank=8, alpha=8, tasks=2):
        super().__init__()
        self.base = base
        self.experts, self.rank = experts, rank
        self.down = nn.Linear(base.in_features, experts * rank, bias=False)  # A_1 ... A_E
        self.up = nn.Linear(experts * rank, base.out_features, bias=False)  # B_1 ... B_E
        nn.init.zeros_(self.up.weight)
        self.routers = nn.ModuleList(nn.Linear(base.in_features, experts, bias=False) for _ in range(tasks))
        self.scale = alpha / rank
        # Per-expert magnitude, initialized to the frozen weight's column norm so W_e' = W at init.
        self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1, keepdim=True).repeat(1, experts))
        self.task = 0

    def forward(self, x):
        """Memory-lean form: never materialize the (out, E, in) effective weight or E full projections.

        Starting from `y = sum_e g_e (m_e / ||W_e||_c) * (W + scale B_e A_e) x`, the per-output factor
        `f_e = m_e / ||W_e||_c` is folded into the expert's B rows, so the experts collapse into one gated
        low-rank matmul, exactly as in MoE-LoRA. Peak activations are ~3 output-sized tensors, not E.
        """
        out_f, in_f = self.base.out_features, self.base.in_features
        gates = self.routers[self.task](x).softmax(dim=-1)  # (N, L, E)
        down = self.down.weight.view(self.experts, self.rank, in_f)  # (E, r, in)
        up = self.up.weight.view(out_f, self.experts, self.rank)  # (out, E, r)
        delta = torch.einsum('oer,eri->oei', self.scale * up, down)  # (out, E, in), transient
        denom = (self.base.weight.unsqueeze(1) + delta).norm(dim=-1) + 1e-6  # (out, E) = ||W_e||_c
        factor = self.magnitude / denom  # (out, E), the DoRA renormalization factor per expert
        base_out = F.linear(x, self.base.weight)  # (N, L, out), no bias
        fused = torch.einsum('nle,oe->nlo', gates, factor).to(base_out.dtype)  # sum_e g_e f_e
        gated = gates.repeat_interleave(self.rank, dim=-1)  # expert e's gate on each of its r hidden units
        delta_out = F.linear(self.down(x) * gated, (factor.unsqueeze(-1) * up).reshape(out_f, -1))
        out = base_out * fused + self.scale * delta_out
        return out if self.base.bias is None else out + self.base.bias  # DINOv3 projections have bias=False


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
    """Route every MoE adapter (MoE-LoRA, MoE-DoRA) inside `module` through the router of `task`."""
    for layer in module.modules():
        if isinstance(layer, (MoELoRA, MoEDoRA)):
            layer.task = task
