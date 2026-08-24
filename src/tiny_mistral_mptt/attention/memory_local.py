from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tiny_mistral.attention.multiresolution import (
    fast_multiresolution_key_value_windows,
    multiresolution_key_indices,
)


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B,H,T,D]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key batch and head dimensions must align")
    if query.shape[-2] != key.shape[-2]:
        raise ValueError("strict-past local attention requires equal Q/K sequence lengths")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")


def strict_past_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    window: int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Exact O(T*W) GQA attention to the previous ``W`` memory positions.

    Query position ``t`` may read keys ``max(0,t-W) .. t-1``. It can never
    read the same-position or a future previous-pass state. Position zero has
    an empty memory set and therefore returns an exact zero vector.
    """
    _validate_qkv(query, key, value)
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0,1)")

    bsz, hq, seq_len, head_dim = query.shape
    hkv = key.shape[1]
    if seq_len == 0:
        return query.clone()
    window = min(int(window), seq_len)
    groups = hq // hkv
    grouped_query = query.reshape(bsz, hkv, groups, seq_len, head_dim)

    # Left-padding by W (rather than W-1 as in ordinary inclusive causal SWA)
    # makes window t contain exactly original memory indices [t-W, ..., t-1].
    padded_key = F.pad(key, (0, 0, window, 0))
    padded_value = F.pad(value, (0, 0, window, 0))
    key_windows = (
        padded_key.unfold(dimension=-2, size=window, step=1)
        .permute(0, 1, 2, 4, 3)[:, :, :seq_len, :, :]
    )
    value_windows = (
        padded_value.unfold(dimension=-2, size=window, step=1)
        .permute(0, 1, 2, 4, 3)[:, :, :seq_len, :, :]
    )

    q_for_mm = grouped_query.permute(0, 1, 3, 2, 4)  # [B,Hkv,T,G,D]
    scores = torch.matmul(q_for_mm, key_windows.transpose(-2, -1))
    scores = scores.permute(0, 1, 3, 2, 4) / math.sqrt(head_dim)  # [B,Hkv,G,T,W]

    t = torch.arange(seq_len, device=query.device)
    j = torch.arange(window, device=query.device)
    valid = (t[:, None] - window + j[None, :]) >= 0
    scores = scores.masked_fill(~valid[None, None, None, :, :], torch.finfo(scores.dtype).min)

    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = probabilities * valid[None, None, None, :, :].to(probabilities.dtype)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        denominator > 0,
        probabilities / denominator.clamp_min(torch.finfo(probabilities.dtype).tiny),
        torch.zeros_like(probabilities),
    )
    if dropout_p:
        probabilities = F.dropout(probabilities, p=dropout_p, training=training)

    probs_for_mm = probabilities.permute(0, 1, 3, 2, 4)  # [B,Hkv,T,G,W]
    output = torch.matmul(probs_for_mm, value_windows)  # [B,Hkv,T,G,D]
    output = output.permute(0, 1, 3, 2, 4).contiguous()
    return output.reshape(bsz, hq, seq_len, head_dim)


def memory_bank_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    memory_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """GQA cross-pass attention into already-strict-past memory records.

    ``memory_mask`` optionally marks valid memory entries with shape ``[B,M]``.
    This is used by sparse cached inference, where different examples may have
    different numbers of committed memories. Empty/all-masked records return an
    exact zero tensor without producing NaNs.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B,H,T,D]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key batch and head dimensions must align")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0,1)")

    bsz, hq, query_len, head_dim = query.shape
    hkv = key.shape[1]
    memory_len = key.shape[-2]
    if memory_mask is not None:
        if memory_mask.shape != (bsz, memory_len) or memory_mask.dtype != torch.bool:
            raise ValueError("memory_mask must be bool [B,M]")
    if memory_len == 0:
        return torch.zeros_like(query)

    groups = hq // hkv
    grouped_query = query.reshape(bsz, hkv, groups, query_len, head_dim)
    grouped_key = key[:, :, None, :, :]  # [B,Hkv,1,M,D]
    grouped_value = value[:, :, None, :, :]  # [B,Hkv,1,M,D]
    scores = torch.matmul(grouped_query, grouped_key.transpose(-2, -1))
    scores = scores / math.sqrt(head_dim)  # [B,Hkv,G,Q,M]

    if memory_mask is None:
        valid = torch.ones((bsz, memory_len), dtype=torch.bool, device=query.device)
    else:
        valid = memory_mask.to(device=query.device)
    scores = scores.masked_fill(
        ~valid[:, None, None, None, :], torch.finfo(scores.dtype).min
    )
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = probabilities * valid[:, None, None, None, :].to(probabilities.dtype)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        denominator > 0,
        probabilities / denominator.clamp_min(torch.finfo(probabilities.dtype).tiny),
        torch.zeros_like(probabilities),
    )
    if dropout_p:
        probabilities = F.dropout(probabilities, p=dropout_p, training=training)

    output = torch.matmul(probabilities, grouped_value)  # [B,Hkv,G,Q,D]
    return output.contiguous().reshape(bsz, hq, query_len, head_dim)


def strict_past_bank_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    writes_before: torch.Tensor,
    memory_mask: torch.Tensor,
    window: int,
    dropout_p: float = 0.0,
    training: bool = False,
    dense: bool = False,
) -> torch.Tensor:
    """O(T*W) GQA cross-pass attention to the last ``W`` committed records.

    ``key``/``value`` are compact chronological memory records ``[B,Hkv,M,D]``.
    ``writes_before[b,t]`` is the number of records committed strictly before
    query position ``t``.  Therefore a memory written at position ``t`` is
    invisible to the query at ``t`` and first becomes visible at ``t+1``.
    ``memory_mask`` is bool ``[B,M]`` for padded compact records. When
    ``dense=True``, the compact records are known to contain exactly one record per
    query position in chronological order, so the ordinary sliding-window
    implementation can be used without the compact-bank gather.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B,H,T,D]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key batch and head dimensions must align")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0,1)")

    bsz, hq, seq_len, head_dim = query.shape
    hkv = key.shape[1]
    memory_len = key.shape[-2]
    if writes_before.shape != (bsz, seq_len):
        raise ValueError("writes_before must be [B,T]")
    if memory_mask.shape != (bsz, memory_len) or memory_mask.dtype != torch.bool:
        raise ValueError("memory_mask must be bool [B,M]")
    if memory_len == 0:
        return torch.zeros_like(query)
    if writes_before.dtype not in (torch.int32, torch.int64):
        raise ValueError("writes_before must have integer dtype")
    if bool((writes_before < 0).any()) or bool((writes_before > memory_len).any()):
        raise ValueError("writes_before is outside compact memory-bank bounds")

    if dense:
        if memory_len != seq_len:
            raise ValueError("dense bank attention requires one memory record per query position")
        if not bool(memory_mask.all()):
            raise ValueError("dense bank attention does not accept padded memory records")
        return strict_past_local_attention(
            query,
            key,
            value,
            window=window,
            dropout_p=dropout_p,
            training=training,
        )

    use_window = min(int(window), memory_len)
    offsets = torch.arange(use_window, device=query.device, dtype=writes_before.dtype)
    # Candidate indices are n-W ... n-1.  Negative indices are invalid padding.
    indices = writes_before[:, :, None] - use_window + offsets[None, None, :]
    valid_index = indices >= 0
    safe_indices = indices.clamp(min=0, max=max(memory_len - 1, 0)).long()

    # Gather compact memory entries into a per-query local bank [B,Hkv,T,W,D].
    key_expanded = key[:, :, None, :, :].expand(-1, -1, seq_len, -1, -1)
    value_expanded = value[:, :, None, :, :].expand(-1, -1, seq_len, -1, -1)
    gather_index = safe_indices[:, None, :, :, None].expand(
        -1, hkv, -1, -1, head_dim
    )
    key_window = torch.gather(key_expanded, dim=3, index=gather_index)
    value_window = torch.gather(value_expanded, dim=3, index=gather_index)

    gathered_mask = torch.gather(memory_mask, dim=1, index=safe_indices.reshape(bsz, -1))
    gathered_mask = gathered_mask.reshape(bsz, seq_len, use_window)
    valid = valid_index & gathered_mask

    groups = hq // hkv
    grouped_query = query.reshape(bsz, hkv, groups, seq_len, head_dim)
    q = grouped_query.permute(0, 1, 3, 2, 4)  # [B,Hkv,T,G,D]
    scores = torch.matmul(q, key_window.transpose(-2, -1))
    scores = scores.permute(0, 1, 3, 2, 4) / math.sqrt(head_dim)
    scores = scores.masked_fill(
        ~valid[:, None, None, :, :], torch.finfo(scores.dtype).min
    )

    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = probabilities * valid[:, None, None, :, :].to(probabilities.dtype)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        denominator > 0,
        probabilities / denominator.clamp_min(torch.finfo(probabilities.dtype).tiny),
        torch.zeros_like(probabilities),
    )
    if dropout_p:
        probabilities = F.dropout(probabilities, p=dropout_p, training=training)

    p = probabilities.permute(0, 1, 3, 2, 4)  # [B,Hkv,T,G,W]
    output = torch.matmul(p, value_window)  # [B,Hkv,T,G,D]
    output = output.permute(0, 1, 3, 2, 4).contiguous()
    return output.reshape(bsz, hq, seq_len, head_dim)


def strict_past_multiscale_bank_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    memory_positions: torch.Tensor,
    memory_mask: torch.Tensor,
    dense_window: int,
    sparse_stride: int,
    sparse_window: int,
    dropout_p: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """One-softmax Memory Attention over dense-recent and sparse-old records.

    The multiscale Memory Attention path writes every previous-pass position. Query ``t`` reads
    ``[t-D,t)`` densely and the last ``S`` periodic positions strictly older
    than that region. The two regions are gathered into one compact bank before
    the GQA score calculation.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B,H,T,D]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key batch and head dimensions must align")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0,1)")

    bsz, hq, query_len, head_dim = query.shape
    hkv = key.shape[1]
    memory_len = key.shape[-2]
    if query_positions.shape != (bsz, query_len):
        raise ValueError("query_positions must be [B,T]")
    if memory_positions.shape != (bsz, memory_len):
        raise ValueError("memory_positions must be [B,M]")
    if memory_mask.shape != (bsz, memory_len) or memory_mask.dtype != torch.bool:
        raise ValueError("memory_mask must be bool [B,M]")
    if memory_len == 0:
        return torch.zeros_like(query)

    # Full-sequence multiscale execution deliberately uses a dense source bank,
    # so absolute linguistic positions are direct gather indices.
    expected = torch.arange(
        memory_len, device=memory_positions.device, dtype=memory_positions.dtype
    )[None, :].expand_as(memory_positions)
    if not torch.equal(memory_positions, expected):
        raise ValueError("multiscale Bank requires one dense record per sequence position")

    if query.device.type == "mps" and query_len == memory_len and torch.equal(query_positions, expected):
        key_bank, value_bank, valid = fast_multiresolution_key_value_windows(
            key,
            value,
            query_positions,
            recent_window=dense_window,
            sparse_stride=sparse_stride,
            sparse_window=sparse_window,
            include_current=False,
            key_padding_mask=memory_mask,
        )
    else:
        indices, index_valid = multiresolution_key_indices(
            query_positions,
            recent_window=dense_window,
            sparse_stride=sparse_stride,
            sparse_window=sparse_window,
            include_current=False,
        )
        compact_width = indices.shape[-1]
        safe_indices = indices.clamp(max=memory_len - 1)
        key_expanded = key[:, :, None, :, :].expand(-1, -1, query_len, -1, -1)
        value_expanded = value[:, :, None, :, :].expand(-1, -1, query_len, -1, -1)
        gather_index = safe_indices[:, None, :, :, None].expand(
            -1, hkv, -1, -1, head_dim
        )
        key_bank = torch.gather(key_expanded, 3, gather_index)
        value_bank = torch.gather(value_expanded, 3, gather_index)
        gathered_mask = torch.gather(
            memory_mask, 1, safe_indices.reshape(bsz, -1)
        ).reshape(bsz, query_len, compact_width)
        valid = index_valid & gathered_mask

    groups = hq // hkv
    grouped_query = query.reshape(bsz, hkv, groups, query_len, head_dim)
    q = grouped_query.permute(0, 1, 3, 2, 4)
    scores = torch.matmul(q, key_bank.transpose(-2, -1))
    scores = scores.permute(0, 1, 3, 2, 4) / math.sqrt(head_dim)
    scores = scores.masked_fill(
        ~valid[:, None, None, :, :], torch.finfo(scores.dtype).min
    )

    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = probabilities * valid[:, None, None, :, :].to(probabilities.dtype)
    denominator = probabilities.sum(dim=-1, keepdim=True)
    probabilities = torch.where(
        denominator > 0,
        probabilities / denominator.clamp_min(torch.finfo(probabilities.dtype).tiny),
        torch.zeros_like(probabilities),
    )
    if dropout_p:
        probabilities = F.dropout(probabilities, p=dropout_p, training=training)

    p = probabilities.permute(0, 1, 3, 2, 4)
    output = torch.matmul(p, value_bank)
    output = output.permute(0, 1, 3, 2, 4).contiguous()
    return output.reshape(bsz, hq, query_len, head_dim)


# Public terminology aliases. The historical function names remain available
# because they are referenced by checkpoints, tests, and downstream scripts.
memory_attention = memory_bank_attention
strict_past_memory_attention = strict_past_bank_attention
strict_past_multiscale_memory_attention = strict_past_multiscale_bank_attention
