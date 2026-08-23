from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .multiresolution import multiresolution_key_indices


def _validate_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B, H, T, D]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key batch and head dimensions must align")
    if query.shape[-2] != key.shape[-2]:
        raise ValueError("local full-sequence attention requires equal Q/K sequence lengths")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")


def local_window_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sliding_window: int | None,
    key_padding_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    sparse_stride: int | None = None,
    sparse_window: int = 0,
) -> torch.Tensor:
    """Exact causal sliding-window attention in O(T * W) score work.

    This backend is intentionally implemented only with ordinary PyTorch tensor
    operations so it can run on CPU and Apple Metal (MPS), where FlexAttention
    is not universally available. It never constructs a T x T score matrix.

    Inputs:
        query: [B, Hq, T, D]
        key:   [B, Hkv, T, D]
        value: [B, Hkv, T, D]

    Mistral v4.45.2 semantics are preserved: ``sliding_window=W`` permits the
    current token plus at most W-1 previous tokens, i.e. q_pos-k_pos < W.

    GQA is handled without physically repeating the K/V sequence across query
    heads. Queries are grouped as [Hkv, Hq/Hkv].
    """
    _validate_qkv(query, key, value)
    if key_padding_mask is not None:
        if key_padding_mask.shape != (query.shape[0], query.shape[-2]) or key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask must be bool [B,T]")
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")

    sparse_window = int(sparse_window)
    if sparse_window < 0:
        raise ValueError("sparse_window must be non-negative")
    if sparse_window:
        if sliding_window is None:
            raise ValueError("sparse attention requires a finite sliding_window")
        if sparse_stride is None or int(sparse_stride) <= 0:
            raise ValueError("sparse attention requires a positive sparse_stride")
        return _multiresolution_local_attention(
            query,
            key,
            value,
            sliding_window=int(sliding_window),
            sparse_stride=int(sparse_stride),
            sparse_window=sparse_window,
            key_padding_mask=key_padding_mask,
            dropout_p=dropout_p,
            training=training,
        )

    bsz, hq, seq_len, head_dim = query.shape
    hkv = key.shape[1]
    if seq_len == 0:
        return query.clone()

    if sliding_window is None:
        window = seq_len
    else:
        window = int(sliding_window)
        if window <= 0:
            raise ValueError("sliding_window must be positive or None")
        window = min(window, seq_len)

    groups = hq // hkv
    # [B, Hkv, G, T, D]
    grouped_query = query.reshape(bsz, hkv, groups, seq_len, head_dim)

    # Left-pad sequence positions so unfold produces, for every query t, the
    # physical window [t-W+1, ..., t]. Padding is zero and is masked before the
    # softmax. `unfold` is a view, so the K/V windows themselves do not make a
    # T*W*D copy here.
    pad_left = window - 1
    padded_key = F.pad(key, (0, 0, pad_left, 0))
    padded_value = F.pad(value, (0, 0, pad_left, 0))

    # Tensor.unfold appends the window dimension at the end:
    # [B, Hkv, T, D, W] -> [B, Hkv, T, W, D].
    key_windows = padded_key.unfold(dimension=-2, size=window, step=1).permute(0, 1, 2, 4, 3)
    value_windows = padded_value.unfold(dimension=-2, size=window, step=1).permute(0, 1, 2, 4, 3)

    # Batched matmul is deliberately used instead of einsum here: it maps to a
    # very well-supported primitive on MPS. The only attention-score tensor is
    # [B, Hkv, G, T, W], so score work/storage is O(T*W), not O(T^2).
    q_for_mm = grouped_query.permute(0, 1, 3, 2, 4)  # [B,Hkv,T,G,D]
    k_for_mm = key_windows.transpose(-2, -1)          # [B,Hkv,T,D,W]
    scores = torch.matmul(q_for_mm, k_for_mm).permute(0, 1, 3, 2, 4)
    scores = scores / math.sqrt(head_dim)

    # Leading windows contain padded positions. A key occupying window slot j
    # for query t corresponds to absolute index t - (W-1-j).
    t = torch.arange(seq_len, device=query.device)
    j = torch.arange(window, device=query.device)
    causal_valid = (t[:, None] - (window - 1 - j[None, :])) >= 0  # [T,W]
    if key_padding_mask is None:
        valid = causal_valid[None, :, :].expand(bsz, -1, -1)
    else:
        padded_valid = F.pad(key_padding_mask, (pad_left, 0), value=False)
        key_valid_windows = padded_valid.unfold(dimension=-1, size=window, step=1)
        valid = causal_valid[None, :, :] & key_valid_windows
    scores = scores.masked_fill(~valid[:, None, None, :, :], torch.finfo(scores.dtype).min)

    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probs = probs * valid[:, None, None, :, :].to(probs.dtype)
    denom = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(
        denom > 0,
        probs / denom.clamp_min(torch.finfo(probs.dtype).tiny),
        torch.zeros_like(probs),
    )
    if dropout_p:
        probs = F.dropout(probs, p=dropout_p, training=training)

    probs_for_mm = probs.permute(0, 1, 3, 2, 4)      # [B,Hkv,T,G,W]
    output = torch.matmul(probs_for_mm, value_windows) # [B,Hkv,T,G,D]
    output = output.permute(0, 1, 3, 2, 4).contiguous()
    return output.reshape(bsz, hq, seq_len, head_dim)


def _multiresolution_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sliding_window: int,
    sparse_stride: int,
    sparse_window: int,
    key_padding_mask: torch.Tensor | None,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Compact O(T * (W+S)) self-attention with one union softmax."""
    bsz, hq, seq_len, head_dim = query.shape
    hkv = key.shape[1]
    if seq_len == 0:
        return query.clone()

    positions = torch.arange(seq_len, device=query.device, dtype=torch.long)
    positions = positions[None, :].expand(bsz, -1)
    indices, index_valid = multiresolution_key_indices(
        positions,
        recent_window=sliding_window,
        sparse_stride=sparse_stride,
        sparse_window=sparse_window,
        include_current=True,
    )
    compact_width = indices.shape[-1]
    safe_indices = indices.clamp(max=seq_len - 1)

    key_expanded = key[:, :, None, :, :].expand(-1, -1, seq_len, -1, -1)
    value_expanded = value[:, :, None, :, :].expand(-1, -1, seq_len, -1, -1)
    gather_index = safe_indices[:, None, :, :, None].expand(
        -1, hkv, -1, -1, head_dim
    )
    key_bank = torch.gather(key_expanded, dim=3, index=gather_index)
    value_bank = torch.gather(value_expanded, dim=3, index=gather_index)

    valid = index_valid
    if key_padding_mask is not None:
        gathered_key_valid = torch.gather(
            key_padding_mask,
            dim=1,
            index=safe_indices.reshape(bsz, -1),
        ).reshape(bsz, seq_len, compact_width)
        valid = valid & gathered_key_valid

    groups = hq // hkv
    grouped_query = query.reshape(bsz, hkv, groups, seq_len, head_dim)
    q_for_mm = grouped_query.permute(0, 1, 3, 2, 4)
    scores = torch.matmul(q_for_mm, key_bank.transpose(-2, -1))
    scores = scores.permute(0, 1, 3, 2, 4) / math.sqrt(head_dim)
    scores = scores.masked_fill(
        ~valid[:, None, None, :, :], torch.finfo(scores.dtype).min
    )

    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probs = probs * valid[:, None, None, :, :].to(probs.dtype)
    denom = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(
        denom > 0,
        probs / denom.clamp_min(torch.finfo(probs.dtype).tiny),
        torch.zeros_like(probs),
    )
    if dropout_p:
        probs = F.dropout(probs, p=dropout_p, training=training)

    probs_for_mm = probs.permute(0, 1, 3, 2, 4)
    output = torch.matmul(probs_for_mm, value_bank)
    output = output.permute(0, 1, 3, 2, 4).contiguous()
    return output.reshape(bsz, hq, seq_len, head_dim)
