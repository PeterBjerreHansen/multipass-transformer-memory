from __future__ import annotations

import torch

from conftest import micro_config
from tiny_mistral.attention import (
    local_window_attention,
    multiresolution_allowed_mask,
    multiresolution_key_indices,
    reference_attention,
)
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.variants.strided_self_attention import StridedSelfAttentionVariant


def test_multiresolution_selector_is_dense_recent_plus_fixed_periodic_old():
    positions = torch.tensor([[9]])
    indices, valid = multiresolution_key_indices(
        positions,
        recent_window=4,
        sparse_stride=3,
        sparse_window=2,
        include_current=True,
    )
    assert valid.tolist() == [[[True, True, True, True, True, True]]]
    assert indices.tolist() == [[[2, 5, 6, 7, 8, 9]]]

    mask = multiresolution_allowed_mask(
        positions,
        torch.arange(10)[None, :],
        recent_window=4,
        sparse_stride=3,
        sparse_window=2,
        include_current=True,
    )
    assert mask[0, 0].nonzero().flatten().tolist() == [2, 5, 6, 7, 8, 9]


def test_compact_strided_self_attention_matches_dense_reference_union_mask():
    torch.manual_seed(81)
    query = torch.randn(2, 4, 11, 8)
    key = torch.randn(2, 2, 11, 8)
    value = torch.randn(2, 2, 11, 8)
    key_valid = torch.ones(2, 11, dtype=torch.bool)
    key_valid[1, 2] = False
    positions = torch.arange(11)[None, :].expand(2, -1)

    compact = local_window_attention(
        query,
        key,
        value,
        sliding_window=4,
        key_padding_mask=key_valid,
        sparse_stride=3,
        sparse_window=2,
    )
    dense = reference_attention(
        query,
        key,
        value,
        query_positions=positions,
        key_positions=positions,
        sliding_window=4,
        key_padding_mask=key_valid,
        sparse_stride=3,
        sparse_window=2,
    )
    torch.testing.assert_close(compact, dense, atol=2e-6, rtol=2e-6)


def test_strided_self_attention_variant_adds_no_parameters_and_only_marks_selected_layers():
    torch.manual_seed(82)
    vanilla_backbone = MistralForCausalLM(micro_config(num_hidden_layers=3))
    torch.manual_seed(82)
    sparse_backbone = MistralForCausalLM(micro_config(num_hidden_layers=3))
    variant = build_variant(
        "strided_self_attention",
        sparse_backbone,
        sparse_attention_stride=3,
        sparse_attention_window=2,
        sparse_attention_layers=[0, 2],
    )
    assert isinstance(variant, StridedSelfAttentionVariant)
    assert variant.sparse_attention_layers == (0, 2)
    assert tuple(variant.state_dict()) == tuple(
        f"backbone.{name}" for name in vanilla_backbone.state_dict()
    )
    assert sum(p.numel() for p in variant.parameters()) == sum(
        p.numel() for p in vanilla_backbone.parameters()
    )
    for index, layer in enumerate(variant.backbone.model.layers):
        expected = 2 if index in {0, 2} else 0
        assert layer.self_attn.sparse_attention_window == expected


def test_strided_self_attention_incremental_cache_matches_full_forward():
    torch.manual_seed(83)
    backbone = MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=4),
        attention_backend="reference",
    ).eval()
    model = StridedSelfAttentionVariant(
        backbone,
        sparse_attention_stride=3,
        sparse_attention_window=2,
        sparse_attention_layers="all",
    ).eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 17))
    with torch.no_grad():
        full = model(ids, use_cache=False).logits
        cache = None
        pieces = []
        for position in range(ids.shape[1]):
            output = model(
                ids[:, position : position + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            assert cache is not None
            pieces.append(output.logits)
            for layer_cache in cache:
                assert layer_cache.positions is not None
                assert layer_cache.seq_len <= 3 + 2
                selected = layer_cache.positions[layer_cache.key_valid]
                assert selected.tolist() == sorted(selected.tolist())
        incremental = torch.cat(pieces, dim=1)
    torch.testing.assert_close(incremental, full, atol=4e-5, rtol=4e-5)
