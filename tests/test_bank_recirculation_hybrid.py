import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.feedback import HybridFeedbackState, HybridPassSource
from tiny_mistral_mptt.inference import exact_decode_step, prefill_exact
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant
from tiny_mistral_mptt.variants.bank_recirculation_hybrid import (
    BankRecirculationHybridVariant,
)


def make_backbone(seed: int = 123) -> MistralForCausalLM:
    torch.manual_seed(seed)
    return MistralForCausalLM(
        micro_config(num_hidden_layers=3, sliding_window=4),
        attention_backend="reference",
    )


def make_hybrid(*, mode: str = "periodic") -> BankRecirculationHybridVariant:
    model = BankRecirculationHybridVariant(
        make_backbone(),
        source_layer=2,
        destination_layer=0,
        alpha=0.1,
        mode="adaptive",
        memory_window=3,
        memory_write_mode=mode,
        memory_write_stride=2,
        memory_token_visibility="visible",
        memory_layers=[1],
        initialization_seed=991,
    )
    return model


def test_zero_bank_readers_reduce_hybrid_to_adaptive_recirculation():
    hybrid = make_hybrid().eval()
    recirculation = RecirculationVariant(
        make_backbone(),
        source_layer=2,
        destination_layer=0,
        alpha=0.1,
        mode="adaptive",
        initialization_seed=991,
    ).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    with torch.no_grad():
        hybrid_output = hybrid.compute_passes(ids, passes=3)
        recirculation_output = recirculation.compute_passes(ids, passes=3)

    for actual, expected in zip(
        hybrid_output.passes, recirculation_output.passes, strict=True
    ):
        torch.testing.assert_close(
            actual.hidden_states, expected.hidden_states, atol=2e-6, rtol=2e-6
        )


def test_pass_source_separates_internal_recurrence_from_top_layer_bank():
    model = make_hybrid().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        run = model._run_first_state(ids)

    assert isinstance(run.feedback_source, HybridPassSource)
    assert run.feedback_source.recurrent_hidden.shape == run.hidden_states.shape
    torch.testing.assert_close(
        run.feedback_source.bank_hidden, run.hidden_states, atol=0, rtol=0
    )
    assert not torch.allclose(
        run.feedback_source.recurrent_hidden, run.feedback_source.bank_hidden
    )


def test_phase_a_trains_bank_reader_and_adaptive_recirculation_controller():
    model = make_hybrid()
    configure_phase(model, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = model.compute_loss(
        ids, phase="A", passes=2, loss_weights=[0.0, 1.0]
    )
    output.loss.backward()

    reader_grad = model.memory_readers["1"].o_proj.weight.grad
    assert reader_grad is not None and reader_grad.abs().sum() > 0
    assert model.adaptive_controller is not None
    controller_grad = model.adaptive_controller.output.weight.grad
    assert controller_grad is not None and controller_grad.abs().sum() > 0
    added = {id(parameter) for parameter in model.added_parameters()}
    for parameter in model.parameters():
        if id(parameter) not in added:
            assert parameter.grad is None


@pytest.mark.parametrize("mode", ["periodic", "memory_token"])
def test_exact_incremental_matches_full_prefix(mode):
    model = make_hybrid(mode=mode).eval()
    with torch.no_grad():
        model.memory_readers["1"].o_proj.weight.copy_(
            torch.eye(model.config.hidden_size)
        )
    if mode == "memory_token":
        vocab = model.config.vocab_size
        ids = torch.tensor([[1, 2, vocab, 3, 4, vocab, 5]])
    else:
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

    prompt_length = 4
    with torch.no_grad():
        state = prefill_exact(model, ids[:, :prompt_length], passes=2)
        for position in range(prompt_length, ids.shape[1] + 1):
            prefix = ids[:, :position]
            full = model.compute_passes(prefix, passes=2)
            expected = model.backbone.lm_head(
                model.prediction_hidden_after_sequence(
                    full.final.hidden_states, prefix
                )
            ).float()[:, -1, :]
            torch.testing.assert_close(
                state.next_token_logits, expected, atol=8e-5, rtol=8e-5
            )
            if position < ids.shape[1]:
                state = exact_decode_step(
                    model, state, ids[:, position : position + 1]
                )

    if mode == "memory_token":
        assert isinstance(state.streams[-1].feedback_memory, HybridFeedbackState)
