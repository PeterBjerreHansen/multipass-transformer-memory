from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .multiresolution import multiresolution_allowed_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand GQA K/V heads to query-head count, matching HF Mistral semantics."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def make_allowed_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    sliding_window: int | None,
    key_padding_mask: torch.Tensor | None = None,
    sparse_stride: int | None = None,
    sparse_window: int = 0,
) -> torch.Tensor:
    """Return [B, Q, K] boolean allowed-attention mask.

    v4.45.2 Mistral convention: key is allowed when it is not in the future and
    q_pos - k_pos < sliding_window. Therefore window=32 permits the current
    position plus 31 previous positions (32 visible keys after warm-up).
    """
    if query_positions.ndim != 2 or key_positions.ndim != 2:
        raise ValueError("query_positions/key_positions must be [B, T]")
    sparse_window = int(sparse_window)
    if sparse_window:
        if sliding_window is None:
            raise ValueError("sparse attention requires a finite sliding_window")
        if sparse_stride is None or int(sparse_stride) <= 0:
            raise ValueError("sparse attention requires a positive sparse_stride")
        return multiresolution_allowed_mask(
            query_positions,
            key_positions,
            recent_window=int(sliding_window),
            sparse_stride=int(sparse_stride),
            sparse_window=sparse_window,
            include_current=True,
            key_padding_mask=key_padding_mask,
        )

    q = query_positions[:, :, None]
    k = key_positions[:, None, :]
    allowed = k <= q
    if sliding_window is not None:
        allowed = allowed & ((q - k) < sliding_window)
    if key_padding_mask is not None:
        if key_padding_mask.shape != key_positions.shape:
            raise ValueError(
                f"key_padding_mask shape {tuple(key_padding_mask.shape)} does not match "
                f"key positions {tuple(key_positions.shape)}"
            )
        allowed = allowed & key_padding_mask[:, None, :].to(torch.bool)
    return allowed


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    sliding_window: int | None,
    key_padding_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    sparse_stride: int | None = None,
    sparse_window: int = 0,
) -> torch.Tensor:
    """Obvious O(QK) correctness implementation; not intended for long training."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B, H, T, D]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query/key batch and head dimensions must align")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")

    n_rep = query.shape[1] // key.shape[1]
    key = repeat_kv(key, n_rep)
    value = repeat_kv(value, n_rep)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    allowed = make_allowed_mask(
        query_positions,
        key_positions,
        sliding_window=sliding_window,
        key_padding_mask=key_padding_mask,
        sparse_stride=sparse_stride,
        sparse_window=sparse_window,
    )
    scores = scores.masked_fill(~allowed[:, None, :, :], torch.finfo(scores.dtype).min)
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    # A write-only control key can make a causal row genuinely empty (for
    # example a leading control slot). Mask and renormalize explicitly so an
    # empty row is exact zero rather than a uniform distribution over masked
    # values caused by softmax(finfo.min, ...).
    valid = allowed[:, None, :, :].to(probs.dtype)
    probs = probs * valid
    denom = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(
        denom > 0,
        probs / denom.clamp_min(torch.finfo(probs.dtype).tiny),
        torch.zeros_like(probs),
    )
    if dropout_p:
        probs = F.dropout(probs, p=dropout_p, training=training)
    return torch.matmul(probs, value)
