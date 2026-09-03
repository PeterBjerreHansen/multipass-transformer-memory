from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..data.packed_dataset import PackedTokenDataset
from ..inference import (
    exact_decode_step,
    prefill_exact,
    recurrent_decode_step,
    recurrent_from_exact,
)
from ..variants.multipass import MultiPassVariant
from .common import (
    NLLAccumulator, block_limit, evaluation_context, packed_evaluation_metadata, source_name,
)


@dataclass(frozen=True)
class RecurrentHorizonResult:
    horizon: int
    exact_nll: float
    recurrent_nll: float
    standard_k1_nll: float
    recurrent_minus_exact: float
    recurrent_minus_standard_k1: float
    predicted_tokens: int
    hidden_delta_rms: float
    hidden_cosine: float


@dataclass(frozen=True)
class RecurrentEvaluationResult:
    prefill_passes: int
    blocks: int
    prompt_tokens: int
    continuation_tokens: int
    predicted_tokens_per_mode: int
    horizons: tuple[RecurrentHorizonResult, ...]
    exact_nll_by_offset: tuple[float, ...]
    recurrent_nll_by_offset: tuple[float, ...]
    standard_k1_nll_by_offset: tuple[float, ...]
    hidden_delta_rms_by_step: tuple[float, ...]
    hidden_cosine_by_step: tuple[float, ...]
    predicted_tokens_by_offset: tuple[int, ...]
    nll_by_source_by_mode: dict[str, dict[str, float]]
    predicted_tokens_by_source: dict[str, int]
    evaluation: dict


def default_horizons(continuation_tokens: int) -> tuple[int, ...]:
    if continuation_tokens < 1:
        raise ValueError("continuation_tokens must be positive")
    values = {1, continuation_tokens}
    horizon = 2
    while horizon < continuation_tokens:
        values.add(horizon)
        horizon *= 2
    return tuple(sorted(values))


def _normalize_horizons(
    horizons: tuple[int, ...] | list[int] | None,
    continuation_tokens: int,
) -> tuple[int, ...]:
    if horizons is None:
        return default_horizons(continuation_tokens)
    result = tuple(sorted({int(value) for value in horizons}))
    if not result:
        raise ValueError("horizons must not be empty")
    if result[0] < 1 or result[-1] > continuation_tokens:
        raise ValueError("horizons must lie in [1, continuation_tokens]")
    return result


@torch.no_grad()
def evaluate_recurrent_continuation(
    model: MultiPassVariant,
    dataset: PackedTokenDataset,
    *,
    device: torch.device | str,
    prefill_passes: int,
    prompt_tokens: int,
    continuation_tokens: int,
    max_blocks: int | None = None,
    horizons: tuple[int, ...] | list[int] | None = None,
    autocast_dtype: str | None = None,
) -> RecurrentEvaluationResult:
    """Compare exact K-stream and collapsed recurrent teacher-forced decoding.

    A pass-1 cached stream from the same checkpoint is evaluated alongside the
    two memory modes as the no-feedback control. Exact and recurrent states share
    one K-pass prefill, so their initial logits (and, for K>1, the first
    processed-token transition) match; later differences measure the
    recurrent train/inference shift directly.
    """
    if prefill_passes < 1:
        raise ValueError("prefill_passes must be positive")
    if prompt_tokens < 1 or continuation_tokens < 1:
        raise ValueError("prompt_tokens and continuation_tokens must be positive")
    if prompt_tokens + continuation_tokens > dataset.sequence_length:
        raise ValueError("prompt + continuation exceeds packed sequence length")
    limit = block_limit(dataset, max_blocks)
    report_horizons = _normalize_horizons(horizons, continuation_tokens)
    scores = {
        name: [NLLAccumulator() for _ in range(continuation_tokens)]
        for name in ("exact", "recurrent", "standard_k1")
    }
    delta_sq_by_step = [0.0] * continuation_tokens
    delta_count_by_step = [0] * continuation_tokens
    cosine_sum_by_step = [0.0] * continuation_tokens
    cosine_count_by_step = [0] * continuation_tokens

    with evaluation_context(model, device=device, autocast_dtype=autocast_dtype):
        for block_index in range(limit):
            ids = dataset.batch([block_index], device=device)
            source = source_name(dataset, block_index) if hasattr(dataset, "manifest") else None
            prompt = ids[:, :prompt_tokens]
            continuation = ids[
                :, prompt_tokens : prompt_tokens + continuation_tokens
            ]

            exact = prefill_exact(model, prompt, passes=prefill_passes)
            recurrent = recurrent_from_exact(exact, decode_mode="feedback")
            standard_k1 = prefill_exact(model, prompt, passes=1)

            for offset in range(continuation_tokens):
                target = continuation[:, offset : offset + 1]
                valid_target = ~model.control_token_mask(target)[:, 0]
                labels = target[:, 0].masked_fill(~valid_target, -100)
                for name, state in (
                    ("exact", exact), ("recurrent", recurrent), ("standard_k1", standard_k1)
                ):
                    scores[name][offset].add(state.next_token_logits, labels, source=source)

                # Process the observed target even at the last offset so the
                # latent drift after every continuation token is measured.
                exact = exact_decode_step(model, exact, target)
                recurrent = recurrent_decode_step(model, recurrent, target)
                standard_k1 = exact_decode_step(model, standard_k1, target)

                delta = recurrent.last_hidden.float() - exact.last_hidden.float()
                delta_sq_by_step[offset] += float(delta.square().sum().cpu())
                delta_count_by_step[offset] += delta.numel()
                cosine = F.cosine_similarity(
                    recurrent.last_hidden.float(),
                    exact.last_hidden.float(),
                    dim=-1,
                )
                identical = delta.float().abs().sum(dim=-1).eq(0)
                cosine = torch.where(identical, torch.ones_like(cosine), cosine)
                cosine_sum_by_step[offset] += float(cosine.sum().cpu())
                cosine_count_by_step[offset] += cosine.numel()
    prediction_count_by_offset = [total.tokens for total in scores["exact"]]

    def offset_means(name: str) -> tuple[float, ...]:
        return tuple(total.mean if total.tokens else float("nan") for total in scores[name])

    exact_offset = offset_means("exact")
    recurrent_offset = offset_means("recurrent")
    standard_k1_offset = offset_means("standard_k1")
    hidden_rms = tuple(
        math.sqrt(total / count)
        for total, count in zip(delta_sq_by_step, delta_count_by_step, strict=True)
    )
    hidden_cosine = tuple(
        total / count
        for total, count in zip(
            cosine_sum_by_step, cosine_count_by_step, strict=True
        )
    )

    horizon_results: list[RecurrentHorizonResult] = []
    for horizon in report_horizons:
        count = sum(prediction_count_by_offset[:horizon])
        if count <= 0:
            raise ValueError("selected horizon contains no linguistic prediction targets")
        exact_nll = sum(total.loss for total in scores["exact"][:horizon]) / count
        recurrent_nll = sum(total.loss for total in scores["recurrent"][:horizon]) / count
        standard_k1_nll = sum(total.loss for total in scores["standard_k1"][:horizon]) / count
        horizon_results.append(
            RecurrentHorizonResult(
                horizon=horizon,
                predicted_tokens=count,
                exact_nll=exact_nll,
                recurrent_nll=recurrent_nll,
                standard_k1_nll=standard_k1_nll,
                recurrent_minus_exact=recurrent_nll - exact_nll,
                recurrent_minus_standard_k1=recurrent_nll - standard_k1_nll,
                hidden_delta_rms=hidden_rms[horizon - 1],
                hidden_cosine=hidden_cosine[horizon - 1],
            )
        )

    by_mode = {}
    for name, offsets in scores.items():
        total = NLLAccumulator()
        for offset in offsets:
            total.merge(offset)
        by_mode[name] = total

    return RecurrentEvaluationResult(
        prefill_passes=prefill_passes,
        blocks=limit,
        prompt_tokens=prompt_tokens,
        continuation_tokens=continuation_tokens,
        predicted_tokens_per_mode=sum(prediction_count_by_offset),
        horizons=tuple(horizon_results),
        exact_nll_by_offset=exact_offset,
        recurrent_nll_by_offset=recurrent_offset,
        standard_k1_nll_by_offset=standard_k1_offset,
        hidden_delta_rms_by_step=hidden_rms,
        hidden_cosine_by_step=hidden_cosine,
        predicted_tokens_by_offset=tuple(prediction_count_by_offset),
        nll_by_source_by_mode={name: total.by_source for name, total in by_mode.items()},
        predicted_tokens_by_source=dict(sorted(by_mode["exact"].source_tokens.items())),
        evaluation=packed_evaluation_metadata(
            model, dataset, device=device, autocast_dtype=autocast_dtype, blocks=limit,
            policy={
                "modes": ["exact_incremental", "feedback", "standard_k1"],
                "prefill_passes": prefill_passes, "standard_prefill_passes": 1,
                "prompt_tokens": prompt_tokens, "continuation_tokens": continuation_tokens,
                "initialization": "data_prefix", "teacher_forced": True,
                "generation": False,
            },
            target_coverage="linguistic_tokens_in_continuation; prompt_unscored; no_added_bos",
        ),
    )
