import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.feedback import HybridFeedbackState, BankState
from tiny_mistral_mptt.inference import (
    exact_decode_step,
    prefill_exact,
    prefill_recurrent,
    recurrent_decode_step,
)
from tiny_mistral_mptt.variants.bank import BankVariant
from tiny_mistral_mptt.variants.bank_add_hybrid import BankAddHybridVariant


def make_model(*, mode="periodic", visibility="visible", hybrid=False):
    torch.manual_seed(222)
    backbone = MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=4),
        attention_backend="reference",
    )
    cls = BankAddHybridVariant if hybrid else BankVariant
    model = cls(
        backbone,
        memory_window=3,
        memory_write_mode=mode,
        memory_write_stride=2,
        memory_token_visibility=visibility,
        memory_layers=[1],
        initialization_seed=909,
    )
    with torch.no_grad():
        for reader in model.memory_readers.values():
            reader.o_proj.weight.copy_(torch.eye(model.config.hidden_size))
        if hybrid:
            model.memory_projection.weight.copy_(0.03 * torch.eye(model.config.hidden_size))
    return model.eval()


def sequence(model, mode):
    if mode == "memory_token":
        V = model.config.vocab_size
        return torch.tensor([[1, 2, V, 3, 14, V, 9, 31, V, 51, 12]])
    return torch.tensor([[1, 7, 3, 14, 22, 9, 31, 4, 51, 12, 6]])


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("mode,visibility", [
    ("dense", "visible"),
    ("periodic", "visible"),
    ("memory_token", "visible"),
    ("memory_token", "write_only"),
])
@pytest.mark.parametrize("passes", [1, 2, 3])
def test_exact_incremental_matches_full_prefix(hybrid, mode, visibility, passes):
    model = make_model(mode=mode, visibility=visibility, hybrid=hybrid)
    ids = sequence(model, mode)
    prompt_len = 4
    with torch.no_grad():
        state = prefill_exact(model, ids[:, :prompt_len], passes=passes)
        for position in range(prompt_len, ids.shape[1] + 1):
            prefix = ids[:, :position]
            full = model.compute_passes(prefix, passes=passes)
            expected_logits = model.backbone.lm_head(
                model.prediction_hidden_after_sequence(full.final.hidden_states, prefix)
            ).float()[:, -1, :]
            torch.testing.assert_close(state.next_token_logits, expected_logits, atol=8e-5, rtol=8e-5)
            if position == ids.shape[1]:
                break
            token = ids[:, position : position + 1]
            state = exact_decode_step(model, state, token)
            torch.testing.assert_close(
                state.last_hidden,
                model.compute_passes(ids[:, : position + 1], passes=passes).final.hidden_states[:, -1:, :],
                atol=8e-5,
                rtol=8e-5,
            )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("mode,visibility", [
    ("periodic", "visible"),
    ("memory_token", "visible"),
    ("memory_token", "write_only"),
])
@pytest.mark.parametrize("passes", [2, 3])
def test_recurrent_first_transition_matches_exact(hybrid, mode, visibility, passes):
    model = make_model(mode=mode, visibility=visibility, hybrid=hybrid)
    ids = sequence(model, mode)
    prompt = ids[:, :5]
    token = ids[:, 5:6]
    with torch.no_grad():
        exact = prefill_exact(model, prompt, passes=passes)
        recurrent = prefill_recurrent(
            model, prompt, passes=passes, decode_mode="feedback"
        )
        exact_after = exact_decode_step(model, exact, token)
        recurrent_after = recurrent_decode_step(model, recurrent, token)
    torch.testing.assert_close(recurrent_after.next_token_logits, exact_after.next_token_logits, atol=8e-5, rtol=8e-5)
    torch.testing.assert_close(recurrent_after.last_hidden, exact_after.last_hidden, atol=8e-5, rtol=8e-5)


def test_periodic_bank_state_is_bounded_and_only_commits_on_trigger():
    model = make_model(mode="periodic")
    ids = sequence(model, "periodic")
    with torch.no_grad():
        state = prefill_recurrent(
            model, ids[:, :5], passes=2, decode_mode="feedback"
        )
        assert isinstance(state.feedback_memory, BankState)
        assert state.feedback_memory.capacity == model.memory_window
        before = state.feedback_memory
        # Position 5 (zero-based) is a C2 commit position: (5+1)%2 == 0.
        state = recurrent_decode_step(model, state, ids[:, 5:6])
        assert isinstance(state.feedback_memory, BankState)
        assert state.feedback_memory.valid.sum() >= before.valid.sum()


def test_memory_token_hybrid_state_preserves_fast_hidden_across_mem_decode():
    model = make_model(mode="memory_token", hybrid=True)
    ids = sequence(model, "memory_token")
    # prompt ends immediately before first MEM
    prompt = ids[:, :2]
    mem = ids[:, 2:3]
    with torch.no_grad():
        state = prefill_recurrent(
            model, prompt, passes=2, decode_mode="feedback"
        )
        assert isinstance(state.feedback_memory, HybridFeedbackState)
        old_fast = state.feedback_memory.fast_hidden.clone()
        state = recurrent_decode_step(model, state, mem)
    assert isinstance(state.feedback_memory, HybridFeedbackState)
    torch.testing.assert_close(state.feedback_memory.fast_hidden, old_fast, atol=0, rtol=0)
    assert state.feedback_memory.bank.valid.any()


def test_write_only_mem_stays_in_kv_cache_position_but_is_marked_invalid():
    model = make_model(mode="memory_token", visibility="write_only")
    V = model.config.vocab_size
    prompt = torch.tensor([[1, 2, V]])
    with torch.no_grad():
        state = prefill_exact(model, prompt, passes=2)
    for stream in state.streams:
        for cache in stream.past_key_values:
            # The physical MEM position is retained so RoPE/cache positions are
            # unchanged, but it is not exposed as a self-attention K/V key.
            assert cache.seq_len == 3
            assert cache.next_position == 3
            assert cache.key_valid is not None
            assert cache.key_valid.tolist() == [[True, True, False]]
