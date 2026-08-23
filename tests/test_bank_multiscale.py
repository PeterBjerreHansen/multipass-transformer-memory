from __future__ import annotations

import torch

from conftest import micro_config
from tiny_mistral.attention.multiresolution import (
    fast_multiresolution_key_value_windows,
    multiresolution_key_indices,
)
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.feedback import BankState
from tiny_mistral_mptt.inference import exact_decode_step, prefill_exact
from tiny_mistral_mptt.variants.bank import BankVariant
from tiny_mistral_mptt.variants.bank_multiscale import MultiscaleBankVariant


def test_fast_multiresolution_windows_match_reference_gather_for_both_causal_modes():
    torch.manual_seed(96)
    batch, heads, sequence, head_dim = 2, 2, 13, 4
    key = torch.randn(batch, heads, sequence, head_dim)
    value = torch.randn_like(key)
    positions = torch.arange(sequence)[None, :].expand(batch, -1)
    mask = torch.ones(batch, sequence, dtype=torch.bool)
    mask[0, 1] = False
    mask[1, 8] = False

    for include_current in (False, True):
        indices, index_valid = multiresolution_key_indices(
            positions,
            recent_window=4,
            sparse_stride=3,
            sparse_window=3,
            include_current=include_current,
        )
        safe_indices = indices.clamp(max=sequence - 1)
        gather_index = safe_indices[:, None, :, :, None].expand(
            -1, heads, -1, -1, head_dim
        )
        expected_key = torch.gather(
            key[:, :, None, :, :].expand(-1, -1, sequence, -1, -1),
            dim=3,
            index=gather_index,
        )
        expected_value = torch.gather(
            value[:, :, None, :, :].expand(-1, -1, sequence, -1, -1),
            dim=3,
            index=gather_index,
        )
        expected_valid = index_valid & torch.gather(
            mask,
            dim=1,
            index=safe_indices.reshape(batch, -1),
        ).reshape(batch, sequence, -1)

        actual_key, actual_value, actual_valid = fast_multiresolution_key_value_windows(
            key,
            value,
            positions,
            recent_window=4,
            sparse_stride=3,
            sparse_window=3,
            include_current=include_current,
            key_padding_mask=mask,
        )
        torch.testing.assert_close(actual_valid, expected_valid, atol=0, rtol=0)
        valid_5d = actual_valid[:, None, :, :, None].expand_as(actual_key)
        torch.testing.assert_close(
            actual_key.masked_select(valid_5d),
            expected_key.masked_select(valid_5d),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            actual_value.masked_select(valid_5d),
            expected_value.masked_select(valid_5d),
            atol=0,
            rtol=0,
        )


def backbone(seed: int) -> MistralForCausalLM:
    torch.manual_seed(seed)
    return MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=4),
        attention_backend="reference",
    )


def activate_readers(model: BankVariant) -> None:
    with torch.no_grad():
        for reader in model.memory_readers.values():
            reader.o_proj.weight.copy_(torch.eye(model.config.hidden_size))


def matching_models(
    *, dense_window: int, sparse_window: int, stride: int
) -> tuple[MultiscaleBankVariant, BankVariant]:
    multiscale = MultiscaleBankVariant(
        backbone(91),
        memory_dense_window=dense_window,
        memory_sparse_window=sparse_window,
        memory_sparse_stride=stride,
        memory_layers=[1],
        initialization_seed=301,
    )
    endpoint = BankVariant(
        backbone(92),
        memory_window=dense_window + sparse_window,
        memory_write_mode="dense" if sparse_window == 0 else "periodic",
        memory_write_stride=1 if sparse_window == 0 else stride,
        memory_layers=[1],
        initialization_seed=302,
    )
    endpoint.load_state_dict(multiscale.state_dict())
    activate_readers(multiscale)
    activate_readers(endpoint)
    return multiscale.eval(), endpoint.eval()


def test_dense_only_endpoint_is_exact_dense_bank():
    multiscale, dense = matching_models(
        dense_window=4, sparse_window=0, stride=3
    )
    ids = torch.tensor([[1, 7, 3, 14, 22, 9, 31, 4, 51, 12]])
    with torch.no_grad():
        for passes in (2, 3):
            actual = multiscale.compute_passes(ids, passes=passes)
            expected = dense.compute_passes(ids, passes=passes)
            torch.testing.assert_close(
                actual.final.hidden_states,
                expected.final.hidden_states,
                atol=0,
                rtol=0,
            )


def test_sparse_only_endpoint_is_exact_periodic_bank():
    multiscale, periodic = matching_models(
        dense_window=0, sparse_window=3, stride=2
    )
    ids = torch.tensor([[1, 7, 3, 14, 22, 9, 31, 4, 51, 12]])
    with torch.no_grad():
        for passes in (2, 3):
            actual = multiscale.compute_passes(ids, passes=passes)
            expected = periodic.compute_passes(ids, passes=passes)
            torch.testing.assert_close(
                actual.final.hidden_states,
                expected.final.hidden_states,
                atol=5e-7,
                rtol=5e-5,
            )


def test_zero_initialized_multiscale_bank_is_exact_vanilla_each_pass():
    model = MultiscaleBankVariant(
        backbone(93),
        memory_dense_window=3,
        memory_sparse_window=2,
        memory_sparse_stride=3,
        memory_layers=[1],
    ).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        outputs = model.compute_passes(ids, passes=3)
    for later in outputs.passes[1:]:
        torch.testing.assert_close(later.logits, outputs.passes[0].logits, atol=0, rtol=0)


def test_multiscale_bank_state_retains_sparse_old_and_dense_recent_positions():
    model = MultiscaleBankVariant(
        backbone(94),
        memory_dense_window=3,
        memory_sparse_window=2,
        memory_sparse_stride=3,
        memory_layers=[1],
    ).eval()
    hidden = torch.randn(1, 10, model.config.hidden_size)
    ids = torch.arange(10)[None, :]
    state = model._feedback_memory_from_hidden(hidden, input_ids=ids)
    assert isinstance(state, BankState)
    assert state.capacity == 5
    assert state.valid.tolist() == [[True, True, True, True, True]]
    assert state.positions.tolist() == [[2, 5, 7, 8, 9]]


def test_multiscale_bank_exact_incremental_matches_full_prefix():
    model = MultiscaleBankVariant(
        backbone(95),
        memory_dense_window=2,
        memory_sparse_window=2,
        memory_sparse_stride=3,
        memory_layers=[1],
        initialization_seed=777,
    ).eval()
    activate_readers(model)
    ids = torch.tensor([[1, 7, 3, 14, 22, 9, 31, 4, 51, 12, 6]])
    prompt_length = 4
    with torch.no_grad():
        state = prefill_exact(model, ids[:, :prompt_length], passes=2)
        for position in range(prompt_length, ids.shape[1] + 1):
            prefix = ids[:, :position]
            full = model.compute_passes(prefix, passes=2)
            expected = model.backbone.lm_head(full.final.hidden_states[:, -1:, :]).float()[:, -1, :]
            torch.testing.assert_close(
                state.next_token_logits, expected, atol=8e-5, rtol=8e-5
            )
            if position < ids.shape[1]:
                state = exact_decode_step(model, state, ids[:, position : position + 1])
