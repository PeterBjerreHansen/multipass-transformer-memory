import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.feedback import HybridFeedbackState, MemoryAttentionState
from tiny_mistral_mptt.inference import (
    exact_decode_step,
    live_feedback_decode_step,
    live_feedback_from_exact,
    prefill_exact_k_pass,
    prefill_live_feedback,
)
from tiny_mistral_mptt.variants.memory_attention import MemoryAttentionVariant
from tiny_mistral_mptt.variants.memory_attention_recurrent_hybrid import MemoryAttentionRecurrentHybridVariant


def make_model(*, mode="periodic", visibility="visible", hybrid=False, stride=2):
    torch.manual_seed(222)
    backbone = MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=4),
        attention_backend="reference",
    )
    cls = MemoryAttentionRecurrentHybridVariant if hybrid else MemoryAttentionVariant
    model = cls(
        backbone,
        **({"recurrent_merger": "projected_residual", "recurrent_layers": [0]} if hybrid else {}),
        memory_window=3,
        memory_write_mode=mode,
        memory_write_stride=stride,
        memory_token_visibility=visibility,
        memory_layers=[1],
        initialization_seed=909,
    )
    with torch.no_grad():
        for reader in model.memory_readers.values():
            reader.o_proj.weight.copy_(torch.eye(model.config.hidden_size))
        if hybrid:
            model.memory_mergers["0"].projection.weight.copy_(0.03 * torch.eye(model.config.hidden_size))
    return model.eval()


def sequence(model, mode):
    if mode == "memory_token":
        V = model.config.vocab_size
        return torch.tensor([[1, 2, V, 3, 14, V, 9, 31, V, 51, 12]])
    return torch.tensor([[1, 7, 3, 14, 22, 9, 31, 4, 51, 12, 6]])


@pytest.mark.parametrize("stride", [2, 3, 4, 5])
def test_periodic_write_origin_counts_a_prefixed_bos_as_a_physical_token(stride):
    model = make_model(stride=stride)
    data = torch.arange(1, 14)[None, :]
    bos = torch.tensor([[model.config.bos_token_id]])
    plain = model.write_mask(data)[0]
    prefixed = model.write_mask(torch.cat((bos, data), dim=1))[0, 1:]
    assert plain.nonzero().flatten().tolist() == list(
        range(stride - 1, data.shape[1], stride)
    )
    assert prefixed.nonzero().flatten().tolist() == list(
        range(stride - 2, data.shape[1], stride)
    )


@pytest.mark.parametrize("stride", [2, 3, 4])
def test_periodic_cached_and_full_paths_agree_for_multiple_strides(stride):
    model = make_model(stride=stride)
    ids = sequence(model, "periodic")[:, :9]
    with torch.no_grad():
        state = prefill_exact_k_pass(model, ids[:, :2], passes=2)
        for position in range(2, ids.shape[1]):
            state = exact_decode_step(model, state, ids[:, position:position + 1])
            prefix = ids[:, : position + 1]
            expected = model.compute_passes(prefix, passes=2).final.logits[:, -1, :]
            fresh = prefill_exact_k_pass(model, prefix, passes=2)
            torch.testing.assert_close(
                state.next_token_logits, expected, atol=8e-5, rtol=8e-5
            )
            for cached_stream, fresh_stream in zip(
                state.streams, fresh.streams, strict=True
            ):
                cached_memory = cached_stream.feedback_memory
                fresh_memory = fresh_stream.feedback_memory
                assert isinstance(cached_memory, MemoryAttentionState)
                assert isinstance(fresh_memory, MemoryAttentionState)
                torch.testing.assert_close(
                    cached_memory.memories,
                    fresh_memory.memories,
                    atol=8e-5,
                    rtol=8e-5,
                )
                torch.testing.assert_close(
                    cached_memory.valid, fresh_memory.valid, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    cached_memory.positions, fresh_memory.positions, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    cached_memory.next_sequence_positions,
                    fresh_memory.next_sequence_positions,
                    atol=0,
                    rtol=0,
                )


@pytest.mark.parametrize("mode,visibility", [
    ("dense", "visible"), ("periodic", "visible"), ("memory_token", "write_only"),
])
def test_k1_conversion_preserves_attention_memory_after_incremental_extension(mode, visibility):
    model = make_model(mode=mode, visibility=visibility)
    ids = sequence(model, mode)
    state = prefill_exact_k_pass(model, ids[:, :1], passes=1)
    for position in range(1, 9):
        state = exact_decode_step(model, state, ids[:, position:position + 1])
        converted = live_feedback_from_exact(state, decode_mode="feedback")
        fresh = prefill_live_feedback(
            model, ids[:, :position + 1], passes=1, decode_mode="feedback"
        )
        token = ids[:, position + 1:position + 2]
        torch.testing.assert_close(
            live_feedback_decode_step(model, converted, token).next_token_logits,
            live_feedback_decode_step(model, fresh, token).next_token_logits,
            atol=8e-5, rtol=8e-5,
        )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("mode,visibility", [
    ("dense", "visible"),
    ("periodic", "visible"),
    ("memory_token", "visible"),
    ("memory_token", "write_only"),
])
@pytest.mark.parametrize("passes", [1, 2, 3])
def test_cached_exact_k_pass_matches_full_prefix(hybrid, mode, visibility, passes):
    model = make_model(mode=mode, visibility=visibility, hybrid=hybrid)
    ids = sequence(model, mode)
    prompt_len = 4
    with torch.no_grad():
        state = prefill_exact_k_pass(model, ids[:, :prompt_len], passes=passes)
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
        exact = prefill_exact_k_pass(model, prompt, passes=passes)
        recurrent = prefill_live_feedback(
            model, prompt, passes=passes, decode_mode="feedback"
        )
        exact_after = exact_decode_step(model, exact, token)
        recurrent_after = live_feedback_decode_step(model, recurrent, token)
    torch.testing.assert_close(recurrent_after.next_token_logits, exact_after.next_token_logits, atol=8e-5, rtol=8e-5)
    torch.testing.assert_close(recurrent_after.last_hidden, exact_after.last_hidden, atol=8e-5, rtol=8e-5)


def test_periodic_memory_state_is_bounded_and_only_commits_on_trigger():
    model = make_model(mode="periodic")
    ids = sequence(model, "periodic")
    with torch.no_grad():
        state = prefill_live_feedback(
            model, ids[:, :5], passes=2, decode_mode="feedback"
        )
        assert isinstance(state.feedback_memory, MemoryAttentionState)
        assert state.feedback_memory.capacity == model.memory_window
        before = state.feedback_memory
        # Position 5 (zero-based) is a C2 commit position: (5+1)%2 == 0.
        state = live_feedback_decode_step(model, state, ids[:, 5:6])
        assert isinstance(state.feedback_memory, MemoryAttentionState)
        assert state.feedback_memory.valid.sum() >= before.valid.sum()


def test_memory_token_hybrid_state_preserves_recurrent_memory_across_mem_decode():
    model = make_model(mode="memory_token", hybrid=True)
    ids = sequence(model, "memory_token")
    # prompt ends immediately before first MEM
    prompt = ids[:, :2]
    mem = ids[:, 2:3]
    with torch.no_grad():
        state = prefill_live_feedback(
            model, prompt, passes=2, decode_mode="feedback"
        )
        assert isinstance(state.feedback_memory, HybridFeedbackState)
        old_memory = state.feedback_memory.recurrent_memory.clone()
        state = live_feedback_decode_step(model, state, mem)
    assert isinstance(state.feedback_memory, HybridFeedbackState)
    torch.testing.assert_close(state.feedback_memory.recurrent_memory, old_memory, atol=0, rtol=0)
    assert state.feedback_memory.memory_attention.valid.any()


def test_write_only_mem_stays_in_kv_cache_position_but_is_marked_invalid():
    model = make_model(mode="memory_token", visibility="write_only")
    V = model.config.vocab_size
    prompt = torch.tensor([[1, 2, V]])
    with torch.no_grad():
        state = prefill_exact_k_pass(model, prompt, passes=2)
    for stream in state.streams:
        for cache in stream.past_key_values:
            # The physical MEM position is retained so RoPE/cache positions are
            # unchanged, but it is not exposed as a self-attention K/V key.
            assert cache.seq_len == 3
            assert cache.next_position == 3
            assert cache.key_valid is not None
            assert cache.key_valid.tolist() == [[True, True, False]]
