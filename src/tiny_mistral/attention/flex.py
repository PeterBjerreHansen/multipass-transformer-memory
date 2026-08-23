from __future__ import annotations

from functools import lru_cache
from typing import Callable

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except Exception:  # pragma: no cover - old torch import path
    create_block_mask = None
    flex_attention = None


@lru_cache(maxsize=64)
def _cached_local_block_mask(
    seq_len: int,
    sliding_window: int | None,
    sparse_stride: int | None,
    sparse_window: int,
    device_type: str,
    device_index: int | None,
    block_size: int,
):
    if create_block_mask is None:
        raise RuntimeError("FlexAttention is unavailable; install PyTorch >= 2.5")
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)

    if sparse_window:
        if sliding_window is None or sparse_stride is None:
            raise ValueError("sparse attention requires finite local window and stride")
        window = int(sliding_window)
        stride = int(sparse_stride)
        count = int(sparse_window)

        def mask_mod(b, h, q_idx, kv_idx):
            age = q_idx - kv_idx
            local = (age >= 0) & (age < window)
            sparse = (
                (age >= window)
                & (age < window + stride * count)
                & ((kv_idx + 1) % stride == 0)
            )
            return local | sparse
    elif sliding_window is None:
        def mask_mod(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx
    else:
        window = int(sliding_window)

        def mask_mod(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & ((q_idx - kv_idx) < window)

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=block_size,
        _compile=(device.type == "cuda"),
    )


@lru_cache(maxsize=4)
def _compiled_flex(dynamic: bool) -> Callable:
    if flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable; install PyTorch >= 2.5")
    return torch.compile(flex_attention, dynamic=dynamic)


def flex_local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sliding_window: int | None,
    key_padding_mask: torch.Tensor | None = None,
    compile_kernel: bool = True,
    block_size: int = 128,
    sparse_stride: int | None = None,
    sparse_window: int = 0,
) -> torch.Tensor:
    """Sparse causal local attention for full, unpadded sequence forwards.

    Query has Hq heads while K/V may have Hkv heads. FlexAttention's native GQA
    support avoids physically repeating K/V heads.
    """
    if flex_attention is None:
        raise RuntimeError("FlexAttention is unavailable; install PyTorch >= 2.5")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be [B, H, T, D]")
    if query.shape[-2] != key.shape[-2] or key.shape != value.shape:
        raise ValueError("flex full-sequence path requires Q/K/V to share sequence length")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by KV head count")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    sparse_window = int(sparse_window)
    if sparse_window < 0:
        raise ValueError("sparse_window must be non-negative")
    if sparse_window and (
        sliding_window is None or sparse_stride is None or int(sparse_stride) <= 0
    ):
        raise ValueError("sparse attention requires finite local window and positive stride")

    seq_len = query.shape[-2]
    device = query.device
    if key_padding_mask is not None:
        if key_padding_mask.shape != (query.shape[0], seq_len) or key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask must be bool [B,T]")
        if create_block_mask is None:
            raise RuntimeError("FlexAttention is unavailable; install PyTorch >= 2.5")
        if sparse_window:
            window = int(sliding_window)
            stride = int(sparse_stride)
            count = sparse_window

            def mask_mod(b, h, q_idx, kv_idx):
                age = q_idx - kv_idx
                local = (age >= 0) & (age < window)
                sparse = (
                    (age >= window)
                    & (age < window + stride * count)
                    & ((kv_idx + 1) % stride == 0)
                )
                return (local | sparse) & key_padding_mask[b, kv_idx]
        elif sliding_window is None:
            def mask_mod(b, h, q_idx, kv_idx):
                return (kv_idx <= q_idx) & key_padding_mask[b, kv_idx]
        else:
            window = int(sliding_window)
            def mask_mod(b, h, q_idx, kv_idx):
                return (kv_idx <= q_idx) & ((q_idx - kv_idx) < window) & key_padding_mask[b, kv_idx]
        block_mask = create_block_mask(
            mask_mod,
            B=query.shape[0],
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            BLOCK_SIZE=block_size,
            _compile=(device.type == "cuda"),
        )
    else:
        block_mask = _cached_local_block_mask(
            seq_len,
            sliding_window,
            sparse_stride,
            sparse_window,
            device.type,
            device.index,
            block_size,
        )
    fn = _compiled_flex(dynamic=False) if compile_kernel else flex_attention
    return fn(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=(query.shape[1] != key.shape[1]),
    )
