from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.packed_dataset import PackedTokenDataset
from .pass_depth import evaluate_pass_depth
from .feedback import FeedbackNLLResult, evaluate_feedback_nll


@dataclass(frozen=True)
class NLLResult:
    nll: float
    perplexity: float
    predicted_tokens: int
    blocks: int
    passes: int
    forward_mode: str
    nll_by_source: dict[str, float]
    predicted_tokens_by_source: dict[str, int]
    evaluation: dict


def evaluate_nll(
    model,
    dataset: PackedTokenDataset,
    *,
    device: torch.device | str,
    passes: int = 1,
    forward_mode: str = "parallel_multipass",
    max_blocks: int | None = None,
    autocast_dtype: str | None = None,
) -> NLLResult | FeedbackNLLResult:
    """Parallel final-pass NLL, or full-block feedback from a single BOS prompt."""
    if forward_mode == "feedback":
        if passes != 1:
            raise ValueError("BOS-only feedback NLL requires passes=1 for prefill")
        return evaluate_feedback_nll(model, dataset, device=device, max_blocks=max_blocks,
                                     autocast_dtype=autocast_dtype)
    if forward_mode != "parallel_multipass":
        raise ValueError("forward_mode must be parallel_multipass or feedback; paper replay was removed")
    result = evaluate_pass_depth(
        model, dataset, device=device, passes=passes, max_blocks=max_blocks,
        autocast_dtype=autocast_dtype,
    )
    return NLLResult(
        nll=result.final_nll,
        perplexity=result.final_perplexity,
        predicted_tokens=result.predicted_tokens,
        blocks=result.blocks,
        passes=result.passes,
        forward_mode=forward_mode,
        nll_by_source=result.nll_by_source_by_pass[-1],
        predicted_tokens_by_source=result.predicted_tokens_by_source,
        evaluation=result.evaluation,
    )
