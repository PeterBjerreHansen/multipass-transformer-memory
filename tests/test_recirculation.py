import pytest
import torch
from torch.nn import functional as F

from tiny_mistral import MistralConfig, MistralForCausalLM
from tiny_mistral_mptt.inference import (
    exact_decode_step,
    prefill_exact_k_pass,
)
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant


def make_model(alpha: float = 0.1, mode: str = "fixed") -> RecirculationVariant:
    torch.manual_seed(0)
    config = MistralConfig(
        vocab_size=31,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=6,
        sliding_window=None,
        use_cache=True,
    )
    backbone = MistralForCausalLM(config)
    return RecirculationVariant(
        backbone,
        source_layer=3,
        destination_layer=1,
        alpha=alpha,
        mode=mode,
    )


def test_first_pass_is_exact_vanilla() -> None:
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    embeddings = model.input_embeddings(input_ids)
    expected = model.backbone.model(
        inputs_embeds=embeddings, use_cache=False
    ).last_hidden_state

    run = model._run_first_state(input_ids)

    torch.testing.assert_close(run.hidden_states, expected, rtol=0, atol=0)


def test_alpha_zero_all_passes_equal_vanilla() -> None:
    model = make_model(alpha=0.0)
    output = model.compute_passes(
        torch.tensor([[1, 2, 3, 4]]), passes=3, phase="B"
    )

    for result in output.passes[1:]:
        torch.testing.assert_close(
            result.hidden_states,
            output.passes[0].hidden_states,
            rtol=0,
            atol=1e-7,
        )


def test_adaptive_controller_starts_at_fixed_mixture() -> None:
    model = make_model(mode="adaptive").eval()
    source = torch.randn(2, 4, model.config.hidden_size)
    destination = torch.randn_like(source)

    with torch.no_grad():
        actual = model._mix(source, destination)
        expected = 0.1 * model._norm_match(source, destination) + 0.9 * destination
        assert model.adaptive_controller is not None
        alpha, beta = model.adaptive_controller(source, destination)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert alpha.shape == source.shape
    assert beta.shape == source.shape
    assert bool(((alpha > 0.0) & (alpha < 1.0)).all())
    assert bool(((beta > 0.0) & (beta < 1.0)).all())
    assert sum(parameter.numel() for parameter in model.added_parameters()) > 0


def test_adaptive_phase_a_trains_only_controller() -> None:
    model = make_model(mode="adaptive")
    configure_phase(model, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    output = model.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()

    added_ids = {id(parameter) for parameter in model.added_parameters()}
    assert any(parameter.grad is not None for parameter in model.added_parameters())
    for parameter in model.backbone.parameters():
        assert parameter.grad is None
    assert all(
        parameter.requires_grad == (id(parameter) in added_ids)
        for parameter in model.parameters()
    )


def test_source_capture_is_source_layer_output_not_final() -> None:
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3]])
    embeddings = model.input_embeddings(input_ids)
    hidden = embeddings
    position_ids = torch.arange(hidden.shape[1])[None, :]
    captured = None
    for index, layer in enumerate(model.backbone.model.layers):
        hidden, _ = layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
            fast_attention_compatible=True,
        )
        if index == model.source_layer:
            captured = hidden

    assert captured is not None
    run = model._run_first_state(input_ids)
    torch.testing.assert_close(run.feedback_source, captured)
    assert not torch.allclose(run.feedback_source, run.hidden_states)


def test_position_zero_is_unmodified_by_feedback() -> None:
    model = make_model(alpha=0.3)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    first = model._run_first_state(input_ids)
    embeddings = model.input_embeddings(input_ids)
    feedback = model._run_feedback_state(input_ids, embeddings, first.feedback_source)

    perturbed = first.feedback_source.clone()
    perturbed[:, 0] *= 1000
    feedback_perturbed = model._run_feedback_state(
        input_ids, embeddings, perturbed
    )

    torch.testing.assert_close(
        feedback.hidden_states[:, 0],
        feedback_perturbed.hidden_states[:, 0],
        atol=1e-7,
        rtol=0,
    )


def test_cached_feedback_memory_is_source_layer_state() -> None:
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    state = prefill_exact_k_pass(model, input_ids, passes=2)
    first = model._run_first_state_cached(input_ids)

    torch.testing.assert_close(
        state.streams[0].feedback_memory,
        first.feedback_source[:, -1:],
    )
    assert not torch.allclose(
        state.streams[0].feedback_memory,
        first.hidden_states[:, -1:],
    )


def test_cached_exact_k_pass_runs_and_advances_source_state() -> None:
    model = make_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    state = prefill_exact_k_pass(model, input_ids, passes=2)
    old_source = state.streams[0].feedback_memory.clone()
    updated = exact_decode_step(model, state, torch.tensor([[5]]))

    assert updated.next_position == state.next_position + 1
    assert updated.streams[0].feedback_memory.shape == old_source.shape
    assert not torch.equal(updated.streams[0].feedback_memory, old_source)


def test_adaptive_cached_exact_k_pass_matches_full_recomputation() -> None:
    model = make_model(mode="adaptive").eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    prompt_length = 4

    with torch.no_grad():
        state = prefill_exact_k_pass(model, ids[:, :prompt_length], passes=3)
        for position in range(prompt_length, ids.shape[1]):
            full = model.compute_passes(ids[:, :position], passes=3)
            torch.testing.assert_close(
                state.next_token_logits,
                full.final.logits[:, -1, :],
                atol=4e-5,
                rtol=4e-5,
            )
            state = exact_decode_step(model, state, ids[:, position : position + 1])
            full_after = model.compute_passes(ids[:, : position + 1], passes=3)
            torch.testing.assert_close(
                state.next_token_logits,
                full_after.final.logits[:, -1, :],
                atol=4e-5,
                rtol=4e-5,
            )


def test_gradients_flow_across_feedback_passes() -> None:
    model = make_model()
    loss = model.compute_loss(
        torch.tensor([[1, 2, 3, 4]]), passes=2, phase="B"
    ).loss
    loss.backward()

    gradients = [parameter.grad for parameter in model.backbone.parameters() if parameter.grad is not None]
    assert gradients and any(float(gradient.abs().sum()) > 0 for gradient in gradients)


@pytest.mark.parametrize(
    "source_layer,destination_layer,alpha",
    [
        (1, 1, 0.1),
        (0, 1, 0.1),
        (5, 1, 0.1),
        (3, 1, -0.1),
        (3, 1, 1.1),
        (3, 1, float("nan")),
    ],
)
def test_invalid_config(source_layer: int, destination_layer: int, alpha: float) -> None:
    config = MistralConfig(
        vocab_size=20,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    backbone = MistralForCausalLM(config)
    with pytest.raises(ValueError):
        RecirculationVariant(
            backbone,
            source_layer=source_layer,
            destination_layer=destination_layer,
            alpha=alpha,
        )
