from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def normalize_pass_weights(
    weights: Sequence[float] | None,
    passes: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return right-aligned non-negative pass weights normalized to sum to one."""
    if passes < 1:
        raise ValueError("passes must be positive")
    if weights is None:
        result = torch.ones(passes, device=device, dtype=dtype)
    else:
        values = [float(value) for value in weights]
        if not values:
            raise ValueError("pass weights must not be empty")
        if len(values) > passes:
            values = values[-passes:]
        elif len(values) < passes:
            values = [0.0] * (passes - len(values)) + values
        result = torch.tensor(values, device=device, dtype=dtype)
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("pass weights must be finite")
    if bool((result < 0).any().item()):
        raise ValueError("pass weights must be non-negative")
    total = result.sum()
    if float(total.detach().cpu()) <= 0:
        raise ValueError("at least one pass weight must be positive")
    return result / total


def causal_lm_loss_from_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy for position-aligned labels.

    Positions with ``ignore_index`` contribute no prediction target, including
    the final position of a packed next-token training block.
    """
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits [B,T,V] and labels [B,T] must align")
    valid = labels.ne(ignore_index)
    if not bool(valid.any()):
        raise ValueError("LM labels contain no prediction targets")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.to(logits.device).reshape(-1),
        ignore_index=ignore_index,
    )


def causal_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Ordinary one-position-shift next-token loss."""
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("logits [B,T,V] and input_ids [B,T] must align")
    if input_ids.shape[1] < 2:
        raise ValueError("causal LM loss requires at least two tokens")
    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:]
    return causal_lm_loss_from_labels(logits, labels)
