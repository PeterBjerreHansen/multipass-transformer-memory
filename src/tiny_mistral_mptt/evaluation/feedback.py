"""Full packed-block NLL using ordinary feedback with one BOS prefill token."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from ..inference import prefill_live_feedback, live_feedback_decode_step
from .common import (
    NLLAccumulator, block_limit, evaluation_context, packed_evaluation_metadata,
    perplexity, source_name,
)


@dataclass(frozen=True)
class FeedbackNLLResult:
    nll: float
    perplexity: float
    predicted_tokens: int
    nll_by_source: dict[str, float]
    predicted_tokens_by_source: dict[str, int]
    aligned_nll: float
    aligned_perplexity: float
    aligned_predicted_tokens: int
    aligned_nll_by_source: dict[str, float]
    aligned_predicted_tokens_by_source: dict[str, int]
    blocks: int
    prefill_passes: int
    forward_mode: str
    evaluation: dict


def feedback_evaluation_metadata(model, dataset, *, device, max_blocks=None, autocast_dtype=None) -> dict:
    """Validate the request before decoding; also serves as report identity."""
    if not getattr(model, "supports_cached_feedback", False):
        raise ValueError("loaded model does not implement feedback decoding")
    bos = model.config.bos_token_id
    if bos is None or not 0 <= bos < model.config.vocab_size:
        raise ValueError("feedback NLL requires a valid model BOS token")
    if getattr(getattr(dataset, "manifest", None), "bos_token_id", bos) != bos:
        raise ValueError("dataset and model BOS token IDs differ")
    # BOS + all but the last physical input consumes exactly L positions.
    # The final target is scored, but need not be consumed to predict another.
    if dataset.sequence_length > model.config.max_position_embeddings:
        raise ValueError("full-block feedback exceeds max_position_embeddings; blocks are not cropped")
    result = packed_evaluation_metadata(
        model, dataset, device=device, autocast_dtype=autocast_dtype,
        blocks=block_limit(dataset, max_blocks),
        policy={
            "forward_mode": "live_feedback", "decode_mode": "feedback",
            "prefill_passes": 1, "prompt_tokens": 1, "prompt_kind": "bos",
            "bos_token_id": bos, "teacher_forced": True,
            "state_reset": "each_block",
            "memory_write_position_basis": "zero_based_physical_sequence_position",
            "periodic_write_rule": "(position + 1) % stride == 0",
            "synthetic_bos_counts_as_physical_token": True,
            "synthetic_bos_shifts_periodic_write_phase": (
                getattr(model, "memory_write_mode", None) == "periodic"
            ),
        },
        target_coverage="all_linguistic_tokens_in_block; added_bos_is_context_only; model_owned_labels",
    )
    result["aligned_target_coverage"] = (
        "exclude_first_linguistic_token_in_each_block; same_targets_as_parallel_nll; added_bos_context"
    )
    result["data"]["consumed_positions_per_block"] = dataset.sequence_length
    return result


def evaluate_feedback_nll(
    model, dataset, *, device, max_blocks: int | None = None,
    autocast_dtype: str | None = None, stop_requested: Callable[[], bool] | None = None,
) -> FeedbackNLLResult:
    metadata = feedback_evaluation_metadata(
        model, dataset, device=device, max_blocks=max_blocks, autocast_dtype=autocast_dtype,
    )
    limit = metadata["data"]["selection"]["stop"]
    scores, aligned = NLLAccumulator(), NLLAccumulator()
    bos_id = metadata["policy"]["bos_token_id"]
    with evaluation_context(model, device=device, autocast_dtype=autocast_dtype):
        for index in range(limit):
            if stop_requested is not None and stop_requested():
                raise InterruptedError("feedback validation interrupted; no partial result committed")
            ids = dataset.batch([index], device=device)
            bos = torch.full((1, 1), bos_id, dtype=ids.dtype, device=ids.device)
            inputs = torch.cat((bos, ids), dim=1)
            labels = model.build_lm_labels(inputs)
            source = source_name(dataset, index) if hasattr(dataset, "manifest") else None
            state = prefill_live_feedback(model, bos, passes=1, decode_mode="feedback")
            # Score at model-owned prediction positions. In MEM mode A predicts
            # B in A <MEM> B; the MEM logits are ignored, but MEM is consumed.
            for position in range(ids.shape[1]):
                if stop_requested is not None and stop_requested():
                    raise InterruptedError("feedback validation interrupted; no partial result committed")
                if position:
                    state = live_feedback_decode_step(
                        model, state, inputs[:, position:position + 1]
                    )
                score = NLLAccumulator()
                score.add(state.next_token_logits, labels[:, position], source=source)
                scores.merge(score)
                if position:
                    aligned.merge(score)
    if not math.isfinite(scores.mean) or not math.isfinite(aligned.mean):
        raise ValueError("feedback validation produced non-finite NLL")
    return FeedbackNLLResult(
        nll=scores.mean, perplexity=perplexity(scores.mean), predicted_tokens=scores.tokens,
        nll_by_source=scores.by_source, predicted_tokens_by_source=dict(scores.source_tokens),
        aligned_nll=aligned.mean, aligned_perplexity=perplexity(aligned.mean),
        aligned_predicted_tokens=aligned.tokens, aligned_nll_by_source=aligned.by_source,
        aligned_predicted_tokens_by_source=dict(aligned.source_tokens),
        blocks=limit, prefill_passes=1, forward_mode="live_feedback", evaluation=metadata,
    )
