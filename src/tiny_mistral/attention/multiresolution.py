from __future__ import annotations

import torch


def _validate_windows(
    *,
    recent_window: int,
    sparse_stride: int,
    sparse_window: int,
) -> tuple[int, int, int]:
    recent_window = int(recent_window)
    sparse_stride = int(sparse_stride)
    sparse_window = int(sparse_window)
    if recent_window < 0 or sparse_window < 0:
        raise ValueError("recent_window and sparse_window must be non-negative")
    if recent_window + sparse_window <= 0:
        raise ValueError("at least one multiresolution window must be non-zero")
    if sparse_stride <= 0:
        raise ValueError("sparse_stride must be positive")
    return recent_window, sparse_stride, sparse_window


def multiresolution_key_indices(
    query_positions: torch.Tensor,
    *,
    recent_window: int,
    sparse_stride: int,
    sparse_window: int,
    include_current: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact chronological key indices and their validity.

    Positions are zero-based and the dense key bank is assumed to contain one
    entry per absolute position. ``include_current=True`` gives ordinary causal
    self-attention semantics: a recent window of W contains the query plus W-1
    preceding keys. ``False`` gives strict-past Bank semantics: the recent
    window contains exactly the preceding W keys.
    """
    recent_window, sparse_stride, sparse_window = _validate_windows(
        recent_window=recent_window,
        sparse_stride=sparse_stride,
        sparse_window=sparse_window,
    )
    if query_positions.ndim != 2 or query_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("query_positions must be integer [B,Q]")
    if bool((query_positions < 0).any()):
        raise ValueError("query_positions must be non-negative")

    query = query_positions[:, :, None]
    pieces: list[torch.Tensor] = []

    if sparse_window:
        # Sparse keys are fixed periodic positions (s+1) % C == 0 and are
        # strictly older than the recent region. ``limit`` is the newest
        # absolute position eligible for sparse retention.
        limit = query - recent_window - (0 if include_current else 1)
        latest = torch.div(limit + 1, sparse_stride, rounding_mode="floor")
        latest = latest * sparse_stride - 1
        offsets = torch.arange(
            sparse_window - 1,
            -1,
            -1,
            device=query_positions.device,
            dtype=query_positions.dtype,
        )
        pieces.append(latest - offsets[None, None, :] * sparse_stride)

    if recent_window:
        if include_current:
            start = query - recent_window + 1
        else:
            start = query - recent_window
        offsets = torch.arange(
            recent_window,
            device=query_positions.device,
            dtype=query_positions.dtype,
        )
        pieces.append(start + offsets[None, None, :])

    indices = torch.cat(pieces, dim=-1)
    valid = indices >= 0
    return indices.clamp_min(0).long(), valid


def multiresolution_allowed_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    recent_window: int,
    sparse_stride: int,
    sparse_window: int,
    include_current: bool,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a boolean [B,Q,K] mask for the same compact policy."""
    recent_window, sparse_stride, sparse_window = _validate_windows(
        recent_window=recent_window,
        sparse_stride=sparse_stride,
        sparse_window=sparse_window,
    )
    if query_positions.ndim != 2 or key_positions.ndim != 2:
        raise ValueError("query_positions/key_positions must be [B,T]")
    if query_positions.shape[0] != key_positions.shape[0]:
        raise ValueError("query/key position batch sizes differ")
    if query_positions.dtype not in (torch.int32, torch.int64) or (
        key_positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("query_positions/key_positions must be integer tensors")

    query = query_positions[:, :, None]
    key = key_positions[:, None, :]
    age = query - key
    minimum_age = 0 if include_current else 1
    recent = (age >= minimum_age) & (age < recent_window + minimum_age)

    if sparse_window:
        limit = query - recent_window - (0 if include_current else 1)
        latest = torch.div(limit + 1, sparse_stride, rounding_mode="floor")
        latest = latest * sparse_stride - 1
        oldest = latest - (sparse_window - 1) * sparse_stride
        sparse = (
            key.ge(oldest)
            & key.le(latest)
            & (key + 1).remainder(sparse_stride).eq(0)
        )
        allowed = recent | sparse
    else:
        allowed = recent

    if key_padding_mask is not None:
        if key_padding_mask.shape != key_positions.shape or key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask must be bool [B,K]")
        allowed = allowed & key_padding_mask[:, None, :]
    return allowed


def retained_multiresolution_indices(
    key_positions: torch.Tensor,
    key_valid: torch.Tensor,
    next_positions: torch.Tensor,
    *,
    recent_window: int,
    sparse_stride: int,
    sparse_window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a bounded strict-past cache for the next query position.

    The input bank may already be sparse, but it must be chronological. Returned
    indices are chronological, padded on the right, and have fixed capacity
    ``recent_window + sparse_window``.
    """
    recent_window, sparse_stride, sparse_window = _validate_windows(
        recent_window=recent_window,
        sparse_stride=sparse_stride,
        sparse_window=sparse_window,
    )
    if key_positions.ndim != 2 or key_valid.shape != key_positions.shape:
        raise ValueError("key_positions/key_valid must share [B,K]")
    if key_valid.dtype != torch.bool:
        raise ValueError("key_valid must be boolean")
    if next_positions.shape != (key_positions.shape[0],):
        raise ValueError("next_positions must be [B]")
    if key_positions.dtype not in (torch.int32, torch.int64) or (
        next_positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("positions must use an integer dtype")

    next_position = next_positions[:, None]
    recent_boundary = next_position - recent_window
    recent = (
        key_valid
        & key_positions.ge(recent_boundary)
        & key_positions.lt(next_position)
    )
    sparse_candidate = (
        key_valid
        & key_positions.lt(recent_boundary)
        & (key_positions + 1).remainder(sparse_stride).eq(0)
    )
    if sparse_window:
        reverse_rank = torch.flip(
            torch.cumsum(torch.flip(sparse_candidate.long(), dims=(1,)), dim=1),
            dims=(1,),
        )
        sparse = sparse_candidate & reverse_rank.le(sparse_window)
    else:
        sparse = torch.zeros_like(sparse_candidate)
    keep = recent | sparse

    capacity = recent_window + sparse_window
    if capacity == 0:  # guarded by _validate_windows; retained for type clarity
        return (
            torch.zeros((key_positions.shape[0], 0), dtype=torch.long, device=key_positions.device),
            torch.zeros((key_positions.shape[0], 0), dtype=torch.bool, device=key_positions.device),
        )
    source = torch.arange(
        key_positions.shape[1], device=key_positions.device, dtype=torch.long
    )[None, :].expand(key_positions.shape[0], -1)
    sentinel = torch.full_like(source, key_positions.shape[1])
    candidates = torch.where(keep, source, sentinel)
    if candidates.shape[1] < capacity:
        candidates = torch.cat(
            (
                candidates,
                torch.full(
                    (key_positions.shape[0], capacity - candidates.shape[1]),
                    key_positions.shape[1],
                    device=key_positions.device,
                    dtype=torch.long,
                ),
            ),
            dim=1,
        )
    ordered = torch.sort(candidates, dim=1).values[:, :capacity]
    selected_valid = ordered.lt(key_positions.shape[1])
    return ordered.clamp(max=max(key_positions.shape[1] - 1, 0)), selected_valid


__all__ = [
    "multiresolution_allowed_mask",
    "multiresolution_key_indices",
    "retained_multiresolution_indices",
]
