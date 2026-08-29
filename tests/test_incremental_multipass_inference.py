import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.inference import (
    exact_decode_step,
    prefill_exact,
    prefill_recurrent,
    recurrent_decode_step,
)
from tiny_mistral_mptt.variants.fbt import FBTVariant
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant


def make_variant(name: str):
    torch.manual_seed(123)
    backbone = MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=4),
        attention_backend="reference",
    )
    if name == "memory_add":
        model = MemoryAddVariant(backbone)
        with torch.no_grad():
            dim = model.config.hidden_size
            model.memory_projection.weight.copy_(0.05 * torch.eye(dim))
    elif name == "fbt":
        model = FBTVariant(backbone, initialization_seed=987)
    else:
        raise AssertionError(name)
    return model.eval()


def sample_ids():
    return torch.tensor([[1, 7, 3, 14, 22, 9, 31, 4, 51, 12, 6, 44, 18]])


def test_fbt_cached_feedback_prefill_is_supported():
    model = make_variant("fbt")
    prompt = sample_ids()[:, :5]
    state = prefill_exact(model, prompt, passes=2)
    assert state.prefill_passes == 2


@pytest.mark.parametrize("variant_name", ["memory_add", "fbt"])
@pytest.mark.parametrize("passes", [1, 2, 3, 4])
def test_exact_incremental_matches_full_recomputation_for_arbitrary_k(
    variant_name, passes
):
    model = make_variant(variant_name)
    ids = sample_ids()
    prompt_length = 5

    with torch.no_grad():
        state = prefill_exact(model, ids[:, :prompt_length], passes=passes)
        full = model.compute_passes(ids[:, :prompt_length], passes=passes)
        torch.testing.assert_close(
            state.next_token_logits,
            full.final.logits[:, -1, :],
            atol=3e-5,
            rtol=3e-5,
        )
        for stream, output in zip(state.streams, full.passes, strict=True):
            torch.testing.assert_close(
                stream.last_hidden,
                output.hidden_states[:, -1:, :],
                atol=3e-5,
                rtol=3e-5,
            )

        # Continue past the SWA window so absolute cache positions and retained
        # W-1 self-attention keys are exercised, not only short-prefix caching.
        for position in range(prompt_length, ids.shape[1]):
            state = exact_decode_step(model, state, ids[:, position : position + 1])
            full = model.compute_passes(ids[:, : position + 1], passes=passes)
            assert state.next_position == position + 1
            torch.testing.assert_close(
                state.next_token_logits,
                full.final.logits[:, -1, :],
                atol=4e-5,
                rtol=4e-5,
            )
            for stream, output in zip(state.streams, full.passes, strict=True):
                torch.testing.assert_close(
                    stream.last_hidden,
                    output.hidden_states[:, -1:, :],
                    atol=4e-5,
                    rtol=4e-5,
                )
                for layer_cache in stream.past_key_values:
                    assert layer_cache.seq_len <= model.config.sliding_window - 1


@pytest.mark.parametrize("variant_name", ["memory_add", "fbt"])
@pytest.mark.parametrize("passes", [2, 3, 4])
def test_recurrent_handoff_is_exact_for_first_processed_token(
    variant_name, passes
):
    model = make_variant(variant_name)
    ids = sample_ids()
    prompt = ids[:, :6]
    first_token = ids[:, 6:7]

    with torch.no_grad():
        exact = prefill_exact(model, prompt, passes=passes)
        recurrent = prefill_recurrent(
            model, prompt, passes=passes, decode_mode="feedback"
        )

        torch.testing.assert_close(
            recurrent.feedback_memory,
            exact.streams[-2].feedback_memory,
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            recurrent.next_token_logits, exact.next_token_logits, atol=0, rtol=0
        )
        torch.testing.assert_close(
            recurrent.last_hidden, exact.last_hidden, atol=0, rtol=0
        )

        exact_after = exact_decode_step(model, exact, first_token)
        recurrent_after = recurrent_decode_step(model, recurrent, first_token)
        torch.testing.assert_close(
            recurrent_after.next_token_logits,
            exact_after.next_token_logits,
            atol=4e-5,
            rtol=4e-5,
        )
        torch.testing.assert_close(
            recurrent_after.last_hidden,
            exact_after.last_hidden,
            atol=4e-5,
            rtol=4e-5,
        )
        assert recurrent_after.next_position == exact_after.next_position == 7


@pytest.mark.parametrize("variant_name", ["memory_add", "fbt"])
def test_k1_standard_decode_and_exact_are_vanilla_cached_inference(variant_name):
    model = make_variant(variant_name)
    ids = sample_ids()
    prompt = ids[:, :5]

    with torch.no_grad():
        vanilla = model.backbone.model(prompt, use_cache=True)
        assert vanilla.past_key_values is not None
        vanilla_logits = model.backbone.lm_head(
            vanilla.last_hidden_state[:, -1:, :]
        ).float()[:, -1, :]

        exact = prefill_exact(model, prompt, passes=1)
        recurrent = prefill_recurrent(
            model, prompt, passes=1, decode_mode="standard"
        )
        assert recurrent.feedback_memory is None
        torch.testing.assert_close(
            exact.next_token_logits, vanilla_logits, atol=0, rtol=0
        )
        torch.testing.assert_close(
            recurrent.next_token_logits, vanilla_logits, atol=0, rtol=0
        )

        vanilla_cache = vanilla.past_key_values
        for position in range(5, ids.shape[1]):
            token = ids[:, position : position + 1]
            vanilla_step = model.backbone.model(
                token, past_key_values=vanilla_cache, use_cache=True
            )
            assert vanilla_step.past_key_values is not None
            vanilla_cache = vanilla_step.past_key_values
            vanilla_logits = model.backbone.lm_head(
                vanilla_step.last_hidden_state
            ).float()[:, -1, :]

            exact = exact_decode_step(model, exact, token)
            recurrent = recurrent_decode_step(model, recurrent, token)
            torch.testing.assert_close(
                exact.next_token_logits, vanilla_logits, atol=0, rtol=0
            )
            torch.testing.assert_close(
                recurrent.next_token_logits, vanilla_logits, atol=0, rtol=0
            )


@pytest.mark.parametrize("variant_name", ["memory_add", "fbt"])
def test_k1_feedback_decode_is_independent_of_prefill_depth(variant_name):
    model = make_variant(variant_name)
    ids = sample_ids()
    prompt = ids[:, :5]

    with torch.no_grad():
        standard = prefill_recurrent(
            model, prompt, passes=1, decode_mode="standard"
        )
        feedback = prefill_recurrent(
            model, prompt, passes=1, decode_mode="feedback"
        )
        assert standard.feedback_memory is None
        assert feedback.feedback_memory is not None
        torch.testing.assert_close(
            standard.next_token_logits, feedback.next_token_logits, atol=0, rtol=0
        )

        token = ids[:, 5:6]
        standard = recurrent_decode_step(model, standard, token)
        feedback = recurrent_decode_step(model, feedback, token)
        assert standard.decode_mode == "standard"
        assert feedback.decode_mode == "feedback"
        assert feedback.feedback_memory is not None
        assert not torch.equal(standard.last_hidden, feedback.last_hidden)


def test_memory_add_recurrent_state_keeps_exactly_one_feedback_vector():
    model = make_variant("memory_add")
    ids = sample_ids()
    with torch.no_grad():
        state = prefill_recurrent(
            model, ids[:, :6], passes=4, decode_mode="feedback"
        )
        assert state.feedback_memory is not None
        assert state.feedback_memory.shape[1] == 1
        for position in range(6, 10):
            state = recurrent_decode_step(model, state, ids[:, position : position + 1])
            assert state.feedback_memory is not None
            assert state.feedback_memory.shape[1] == 1
            torch.testing.assert_close(
                state.feedback_memory, state.last_hidden, atol=0, rtol=0
            )
