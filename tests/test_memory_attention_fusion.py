import pytest
import torch

from tiny_mistral_mptt.variants.memory_attention_fusion import MemoryAttentionFusion


def fusion(mode: str) -> MemoryAttentionFusion:
    return MemoryAttentionFusion(
        4,
        mode=mode,
        controller_hidden_size=None if mode == "residual" else 3,
        initialization_seed=17,
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("residual", lambda d, a: d + a),
        ("destination_gated", lambda d, a: 0.9 * d + a),
        ("attention_gated", lambda d, a: d + 0.1 * a),
        ("dual_gated", lambda d, a: 0.9 * d + 0.1 * a),
    ],
)
def test_fusion_initialization_implements_the_four_equations(mode, expected):
    destination = torch.randn(2, 3, 4)
    attention = torch.randn_like(destination)
    available = torch.ones(2, 3, dtype=torch.bool)
    actual = fusion(mode)(destination, attention, available=available)
    torch.testing.assert_close(actual, expected(destination, attention))


@pytest.mark.parametrize(
    ("mode", "alpha", "beta"),
    [
        ("residual", None, None),
        ("destination_gated", None, 0.9),
        ("attention_gated", 0.1, None),
        ("dual_gated", 0.1, 0.9),
    ],
)
def test_gate_shapes_and_independent_initial_values(mode, alpha, beta):
    destination = torch.randn(2, 3, 4)
    attention = torch.randn_like(destination)
    gates = fusion(mode).gate_values(attention, destination)
    for actual, expected in ((gates.alpha, alpha), (gates.beta, beta)):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None and actual.shape == destination.shape
            torch.testing.assert_close(actual, torch.full_like(actual, expected))


@pytest.mark.parametrize(
    "mode",
    ["residual", "destination_gated", "attention_gated", "dual_gated"],
)
def test_unavailable_positions_are_an_exact_bypass(mode):
    destination = torch.randn(2, 3, 4)
    attention = torch.randn_like(destination)
    available = torch.tensor([[False, True, False], [True, False, True]])
    actual = fusion(mode)(destination, attention, available=available)
    torch.testing.assert_close(
        actual[~available], destination[~available], atol=0, rtol=0
    )


def test_single_gate_arms_have_equal_parameter_counts_and_dual_adds_one_head():
    destination = fusion("destination_gated")
    attention = fusion("attention_gated")
    dual = fusion("dual_gated")
    residual = fusion("residual")
    counts = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in {
            "destination": destination,
            "attention": attention,
            "dual": dual,
            "residual": residual,
        }.items()
    }
    assert counts["residual"] == 0
    assert counts["destination"] == counts["attention"]
    assert counts["dual"] - counts["attention"] == 4 * (3 + 1)


def test_gate_output_learns_on_first_step_and_trunk_on_second_step():
    module = fusion("dual_gated")
    assert module.controller is not None
    destination = torch.randn(2, 3, 4)
    attention = torch.randn_like(destination)
    available = torch.ones(2, 3, dtype=torch.bool)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    module(destination, attention, available=available).square().mean().backward()
    assert module.controller.output.weight.grad is not None
    assert module.controller.output.weight.grad.abs().sum() > 0
    assert module.controller.hidden_1.weight.grad is not None
    assert module.controller.hidden_1.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    module(destination, attention, available=available).square().mean().backward()
    assert module.controller.hidden_1.weight.grad is not None
    assert module.controller.hidden_1.weight.grad.abs().sum() > 0


def test_gated_modes_require_width_and_residual_forbids_it():
    with pytest.raises(ValueError, match="requires a positive"):
        MemoryAttentionFusion(
            4,
            mode="attention_gated",
            controller_hidden_size=None,
            initialization_seed=1,
        )
    with pytest.raises(ValueError, match="not used"):
        MemoryAttentionFusion(
            4,
            mode="residual",
            controller_hidden_size=3,
            initialization_seed=1,
        )
