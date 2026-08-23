import math

import torch

from tiny_mistral_mptt.attention.memory_local import strict_past_local_attention


def dense_reference(query, key, value, window):
    bsz, hq, seq_len, dim = query.shape
    hkv = key.shape[1]
    groups = hq // hkv
    key = (
        key[:, :, None, :, :]
        .expand(bsz, hkv, groups, seq_len, dim)
        .reshape(bsz, hq, seq_len, dim)
    )
    value = (
        value[:, :, None, :, :]
        .expand(bsz, hkv, groups, seq_len, dim)
        .reshape(bsz, hq, seq_len, dim)
    )
    scores = query @ key.transpose(-2, -1) / math.sqrt(dim)
    q = torch.arange(seq_len)[:, None]
    k = torch.arange(seq_len)[None, :]
    allowed = (k < q) & ((q - k) <= window)
    scores = scores.masked_fill(~allowed[None, None], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    probs = probs * allowed[None, None]
    denom = probs.sum(-1, keepdim=True)
    probs = torch.where(
        denom > 0,
        probs / denom.clamp_min(torch.finfo(probs.dtype).tiny),
        torch.zeros_like(probs),
    )
    return probs @ value


def test_strict_past_local_attention_matches_dense_gqa_reference():
    torch.manual_seed(4)
    q = torch.randn(2, 4, 9, 8)
    k = torch.randn(2, 2, 9, 8)
    v = torch.randn(2, 2, 9, 8)
    actual = strict_past_local_attention(q, k, v, window=4)
    expected = dense_reference(q, k, v, window=4)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual[:, :, 0], torch.zeros_like(actual[:, :, 0]), atol=0, rtol=0)


def test_memory_attention_ignores_current_future_and_too_old_memory():
    torch.manual_seed(5)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn(1, 1, 8, 4)
    v = torch.randn(1, 1, 8, 4)
    baseline = strict_past_local_attention(q, k, v, window=3)

    # Query 6 can read only memory positions 3,4,5.
    changed = v.clone()
    changed[:, :, 0] += 1000  # too old
    changed[:, :, 6:] -= 1000  # current/future
    perturbed = strict_past_local_attention(q, k, changed, window=3)
    torch.testing.assert_close(perturbed[:, :, 6], baseline[:, :, 6], atol=0, rtol=0)


def dense_bank_reference(query, key, value):
    bsz, hq, query_len, dim = query.shape
    hkv = key.shape[1]
    memory_len = key.shape[-2]
    if memory_len == 0:
        return torch.zeros_like(query)
    groups = hq // hkv
    key = (
        key[:, :, None, :, :]
        .expand(bsz, hkv, groups, memory_len, dim)
        .reshape(bsz, hq, memory_len, dim)
    )
    value = (
        value[:, :, None, :, :]
        .expand(bsz, hkv, groups, memory_len, dim)
        .reshape(bsz, hq, memory_len, dim)
    )
    scores = query @ key.transpose(-2, -1) / math.sqrt(dim)
    return torch.softmax(scores, dim=-1) @ value


def test_memory_bank_attention_matches_dense_gqa_reference():
    from tiny_mistral_mptt.attention.memory_local import memory_bank_attention

    torch.manual_seed(6)
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 8)
    actual = memory_bank_attention(q, k, v)
    expected = dense_bank_reference(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_memory_bank_attention_matches_strict_past_last_query():
    from tiny_mistral_mptt.attention.memory_local import memory_bank_attention

    torch.manual_seed(7)
    q = torch.randn(1, 4, 9, 8)
    k = torch.randn(1, 2, 9, 8)
    v = torch.randn(1, 2, 9, 8)
    strict = strict_past_local_attention(q, k, v, window=4)
    bank = memory_bank_attention(
        q[:, :, -1:, :],
        k[:, :, -5:-1, :],
        v[:, :, -5:-1, :],
    )
    torch.testing.assert_close(bank[:, :, 0], strict[:, :, -1], atol=1e-6, rtol=1e-6)


def test_memory_bank_attention_empty_bank_is_exact_zero():
    from tiny_mistral_mptt.attention.memory_local import memory_bank_attention

    q = torch.randn(2, 4, 1, 8)
    k = torch.empty(2, 2, 0, 8)
    v = torch.empty(2, 2, 0, 8)
    actual = memory_bank_attention(q, k, v)
    torch.testing.assert_close(actual, torch.zeros_like(q), atol=0, rtol=0)


def test_masked_memory_bank_returns_zero_for_empty_rows_without_nan():
    from tiny_mistral_mptt.attention.memory_local import memory_bank_attention

    torch.manual_seed(2)
    q = torch.randn(2, 4, 1, 3)
    k = torch.randn(2, 2, 4, 3)
    v = torch.randn(2, 2, 4, 3)
    mask = torch.tensor([[False, False, False, False], [True, False, True, False]])
    out = memory_bank_attention(q, k, v, memory_mask=mask)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0], torch.zeros_like(out[0]), atol=0, rtol=0)
    assert out[1].abs().sum() > 0


def test_bank_attention_c1_matches_strict_local_attention():
    from tiny_mistral_mptt.attention.memory_local import (
        strict_past_local_attention,
        strict_past_bank_attention,
    )

    torch.manual_seed(3)
    q = torch.randn(1, 4, 7, 3)
    k = torch.randn(1, 2, 7, 3)
    v = torch.randn(1, 2, 7, 3)
    writes_before = torch.arange(7)[None, :]
    mask = torch.ones(1, 7, dtype=torch.bool)
    dense = strict_past_local_attention(q, k, v, window=4)
    bank = strict_past_bank_attention(
        q, k, v, writes_before=writes_before, memory_mask=mask, window=4
    )
    torch.testing.assert_close(bank, dense, atol=0, rtol=0)


def test_bank_attention_window_counts_records_not_source_distance():
    from tiny_mistral_mptt.attention.memory_local import strict_past_bank_attention

    # One query head/KV head and scalar head dimension makes the retention test
    # easy to audit. Query t=5 has four committed records, but W=2 means only
    # records 2 and 3 may affect it.
    q = torch.ones(1, 1, 6, 1)
    k = torch.zeros(1, 1, 4, 1)
    v = torch.tensor([[[[100.0], [200.0], [3.0], [5.0]]]])
    writes_before = torch.tensor([[0, 1, 1, 2, 3, 4]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    out = strict_past_bank_attention(
        q, k, v, writes_before=writes_before, memory_mask=mask, window=2
    )
    # Equal keys => uniform attention over the two retained records: (3+5)/2.
    torch.testing.assert_close(out[0, 0, 5, 0], torch.tensor(4.0), atol=0, rtol=0)
