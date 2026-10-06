"""Adapters: exact no-ops at init, only adapter weights train, MoE-LoRA routes per task."""
import pytest
import torch
from torch import nn

from sarbench.adapters import DoRA, LoRA, MoEDoRA, MoELoRA, add_adapters, set_task
from sarbench.backbones import build_backbone

# 12 blocks x rank 16 x ((768 + 2304) + (768 + 768)) for A and B of qkv and proj;
# DoRA adds the per-output magnitude (2304 + 768 per block); MoE-LoRA / MoE-DoRA: 4 experts x rank 4 = 16,
# two routers per adapter, and MoE-DoRA also the per-expert magnitudes (4 x (2304 + 768) per block)
TRAINABLE = {'lora': 884_736, 'dora': 921_600, 'moelora': 1_032_192, 'moedora': 1_179_648}
ADAPTER_WEIGHTS = ('.down.weight', '.up.weight', '.magnitude', '.routers.0.weight', '.routers.1.weight')


def tokens(backbone, x):
    """Tokens without gradients on the GPU; .cuda() also moves adapters added since the last call."""
    with torch.no_grad():
        return backbone.cuda().eval()(x)


def tokens_per_task(name, kind):
    """Tokens under task 0 and task 1, after giving every adapter a non-zero B."""
    backbone = build_backbone(name, pretrained=False)
    add_adapters(backbone, kind)
    for layer in backbone.modules():
        if isinstance(layer, (DoRA, LoRA, MoELoRA, MoEDoRA)):
            nn.init.normal_(layer.up.weight, std=0.02)
    x = torch.randn(2, 1, 512, 512, device='cuda')
    set_task(backbone, 0)
    task0 = tokens(backbone, x)
    set_task(backbone, 1)
    return task0, tokens(backbone, x)


@pytest.mark.parametrize('kind,tol', [('lora', 1e-6), ('dora', 1e-5), ('moelora', 1e-6), ('moedora', 1e-5)])
@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_adapters_are_noops_at_init(name, kind, tol):
    backbone = build_backbone(name, pretrained=True)
    x = torch.randn(2, 1, 512, 512, device='cuda')
    before = tokens(backbone, x)
    add_adapters(backbone, kind)
    assert (tokens(backbone, x) - before).abs().max() <= tol  # DoRA's weight renormalization adds fp noise


@pytest.mark.parametrize('kind', ['lora', 'dora', 'moelora', 'moedora'])
@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_only_adapter_weights_train(name, kind):
    backbone = build_backbone(name, pretrained=False)
    add_adapters(backbone, kind)
    trainable = {n: p.numel() for n, p in backbone.named_parameters() if p.requires_grad}
    assert all(n.endswith(ADAPTER_WEIGHTS) for n in trainable)
    assert sum(trainable.values()) == TRAINABLE[kind]


@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_moelora_tokens_depend_on_task(name):
    task0, task1 = tokens_per_task(name, 'moelora')
    assert (task0 - task1).abs().max() > 1e-3


@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_moedora_tokens_depend_on_task(name):
    task0, task1 = tokens_per_task(name, 'moedora')
    assert (task0 - task1).abs().max() > 1e-3


@pytest.mark.parametrize('name', ['vit', 'terramind'])
def test_lora_tokens_do_not_depend_on_task(name):
    task0, task1 = tokens_per_task(name, 'lora')
    assert torch.equal(task0, task1)


def test_add_adapters_respects_the_configured_target_groups():
    """The YAML's peft.targets selects which module groups a backbone exposes for adaptation."""
    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 8)
            self.mlp = nn.Linear(8, 8)
            self.embed_dim = 8

        def adapter_targets(self, groups=None):
            groups = groups or ('attn',)
            targets = []
            if 'attn' in groups:
                targets.append((self, 'qkv'))
            if 'mlp' in groups:
                targets.append((self, 'mlp'))
            return targets

    backbone = Stub()
    add_adapters(backbone, 'lora', targets=['attn', 'mlp'])
    assert isinstance(backbone.qkv, LoRA) and isinstance(backbone.mlp, LoRA)
    assert not backbone.qkv.base.weight.requires_grad  # the frozen base stays frozen
    assert backbone.qkv.down.weight.requires_grad


def test_dora_is_a_noop_at_init_and_trains_its_magnitude():
    base = nn.Linear(16, 8)
    layer = DoRA(base, rank=4, alpha=8).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    torch.testing.assert_close(layer(x), base(x))  # B = 0 and m = ||W||_c -> W' = W exactly
    nn.init.normal_(layer.up.weight, std=0.1)
    assert not torch.allclose(layer(x), base(x))
    assert layer.magnitude.requires_grad and layer.down.weight.requires_grad


def test_moedora_is_a_noop_at_init_and_routes_by_task():
    base = nn.Linear(16, 8)
    layer = MoEDoRA(base, experts=3, rank=2, alpha=4).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64)
    torch.testing.assert_close(layer(x), base(x))  # B_e = 0, m_e = ||W||_c -> every expert equals W
    nn.init.normal_(layer.up.weight, std=0.1)
    assert not torch.allclose(layer(x), base(x))
    assert layer.magnitude.requires_grad and layer.down.weight.requires_grad


def test_moelora_is_a_gated_sum_of_lora_experts():
    """The stacked down/up layers compute y = W x + (alpha / r) * sum_e g_e(x) B_e A_e x."""
    # 3 experts of rank 2 (unequal, so mixing up the two axes fails); alpha / r = 2
    layer = MoELoRA(nn.Linear(32, 24), experts=3, rank=2, alpha=4).double()
    nn.init.normal_(layer.up.weight)
    set_task(layer, 1)
    x = torch.randn(2, 5, 32, dtype=torch.float64)  # float64: unaffected by reduced-precision fp32 matmul settings
    gates = layer.routers[1](x).softmax(dim=-1)
    A = layer.down.weight.split(2)  # A_e: [2, 32]
    B = layer.up.weight.split(2, dim=1)  # B_e: [24, 2]
    experts = sum(gates[..., e, None] * (x @ A[e].T @ B[e].T) for e in range(3))
    torch.testing.assert_close(layer(x), layer.base(x) + 2 * experts)
