"""Teacher-forced exact-K-pass versus live-feedback continuation diagnostic."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..data.packed_dataset import PackedTokenDataset
from ..inference import (
    exact_decode_step,
    live_feedback_decode_step,
    live_feedback_from_exact,
    prefill_exact_k_pass,
)
from ..variants.multipass import MultiPassVariant
from .common import (
    NLLAccumulator,
    block_limit,
    evaluation_context,
    packed_evaluation_metadata,
    source_name,
)


@dataclass(frozen=True)
class FeedbackHorizonResult:
    horizon: int
    exact_k_pass_nll: float
    live_feedback_nll: float
    standard_k1_nll: float
    live_feedback_minus_exact_k_pass: float
    live_feedback_minus_standard_k1: float
    retained_k4_improvement: float | None
    predicted_tokens: int
    exact_to_live_kl: float
    top1_agreement: float
    hidden_delta_rms: float
    hidden_cosine: float


@dataclass(frozen=True)
class FeedbackEvaluationResult:
    prefill_passes: int
    blocks: int
    prompt_tokens: int
    continuation_tokens: int
    predicted_tokens_per_mode: int
    horizons: tuple[FeedbackHorizonResult, ...]
    exact_k_pass_nll_by_offset: tuple[float, ...]
    live_feedback_nll_by_offset: tuple[float, ...]
    standard_k1_nll_by_offset: tuple[float, ...]
    exact_to_live_kl_by_offset: tuple[float, ...]
    top1_agreement_by_offset: tuple[float, ...]
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
def evaluate_feedback_continuation(
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
) -> FeedbackEvaluationResult:
    """Compare exact K-stream and collapsed live-feedback decoding."""
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
        for name in ("exact_k_pass", "live_feedback", "standard_k1")
    }
    delta_sq_by_step = [0.0] * continuation_tokens
    delta_count_by_step = [0] * continuation_tokens
    cosine_sum_by_step = [0.0] * continuation_tokens
    cosine_count_by_step = [0] * continuation_tokens
    kl_sum_by_offset = [0.0] * continuation_tokens
    top1_equal_by_offset = [0] * continuation_tokens
    distribution_count_by_offset = [0] * continuation_tokens

    with evaluation_context(model, device=device, autocast_dtype=autocast_dtype):
        for block_index in range(limit):
            ids = dataset.batch([block_index], device=device)
            source = (
                source_name(dataset, block_index)
                if hasattr(dataset, "manifest")
                else None
            )
            prompt = ids[:, :prompt_tokens]
            continuation = ids[
                :, prompt_tokens : prompt_tokens + continuation_tokens
            ]

            exact = prefill_exact_k_pass(model, prompt, passes=prefill_passes)
            live = live_feedback_from_exact(exact, decode_mode="feedback")
            standard_k1 = prefill_exact_k_pass(model, prompt, passes=1)

            for offset in range(continuation_tokens):
                target = continuation[:, offset : offset + 1]
                valid_target = ~model.control_token_mask(target)[:, 0]
                labels = target[:, 0].masked_fill(~valid_target, -100)
                for name, state in (
                    ("exact_k_pass", exact),
                    ("live_feedback", live),
                    ("standard_k1", standard_k1),
                ):
                    scores[name][offset].add(
                        state.next_token_logits, labels, source=source
                    )

                exact_log_probs = F.log_softmax(
                    exact.next_token_logits.float(), dim=-1
                )
                live_log_probs = F.log_softmax(
                    live.next_token_logits.float(), dim=-1
                )
                per_example_kl = (
                    exact_log_probs.exp()
                    * (exact_log_probs - live_log_probs)
                ).sum(dim=-1)
                kl_sum_by_offset[offset] += float(
                    per_example_kl[valid_target].sum().cpu()
                )
                top1_equal_by_offset[offset] += int(
                    (
                        exact.next_token_logits.argmax(dim=-1)
                        == live.next_token_logits.argmax(dim=-1)
                    )[valid_target]
                    .sum()
                    .cpu()
                )
                distribution_count_by_offset[offset] += int(valid_target.sum().cpu())

                exact = exact_decode_step(model, exact, target)
                live = live_feedback_decode_step(model, live, target)
                standard_k1 = exact_decode_step(model, standard_k1, target)

                delta = live.last_hidden.float() - exact.last_hidden.float()
                delta_sq_by_step[offset] += float(delta.square().sum().cpu())
                delta_count_by_step[offset] += delta.numel()
                cosine = F.cosine_similarity(
                    live.last_hidden.float(), exact.last_hidden.float(), dim=-1
                )
                identical = delta.abs().sum(dim=-1).eq(0)
                cosine = torch.where(identical, torch.ones_like(cosine), cosine)
                cosine_sum_by_step[offset] += float(cosine.sum().cpu())
                cosine_count_by_step[offset] += cosine.numel()

    prediction_count_by_offset = [
        total.tokens for total in scores["exact_k_pass"]
    ]

    def offset_means(name: str) -> tuple[float, ...]:
        return tuple(
            total.mean if total.tokens else float("nan") for total in scores[name]
        )

    exact_offset = offset_means("exact_k_pass")
    live_offset = offset_means("live_feedback")
    standard_k1_offset = offset_means("standard_k1")
    kl_offset = tuple(
        total / count if count else float("nan")
        for total, count in zip(
            kl_sum_by_offset, distribution_count_by_offset, strict=True
        )
    )
    top1_offset = tuple(
        equal / count if count else float("nan")
        for equal, count in zip(
            top1_equal_by_offset, distribution_count_by_offset, strict=True
        )
    )
    hidden_rms = tuple(
        math.sqrt(total / count)
        for total, count in zip(
            delta_sq_by_step, delta_count_by_step, strict=True
        )
    )
    hidden_cosine = tuple(
        total / count
        for total, count in zip(
            cosine_sum_by_step, cosine_count_by_step, strict=True
        )
    )

    horizon_results: list[FeedbackHorizonResult] = []
    for horizon in report_horizons:
        count = sum(prediction_count_by_offset[:horizon])
        if count <= 0:
            raise ValueError("selected horizon contains no linguistic prediction targets")
        exact_nll = (
            sum(total.loss for total in scores["exact_k_pass"][:horizon]) / count
        )
        live_nll = (
            sum(total.loss for total in scores["live_feedback"][:horizon]) / count
        )
        standard_k1_nll = (
            sum(total.loss for total in scores["standard_k1"][:horizon]) / count
        )
        improvement = standard_k1_nll - exact_nll
        retained = (
            None
            if prefill_passes != 4 or improvement <= 0.0
            else (standard_k1_nll - live_nll) / improvement
        )
        distribution_count = sum(distribution_count_by_offset[:horizon])
        hidden_count = sum(delta_count_by_step[:horizon])
        cosine_count = sum(cosine_count_by_step[:horizon])
        horizon_results.append(
            FeedbackHorizonResult(
                horizon=horizon,
                predicted_tokens=count,
                exact_k_pass_nll=exact_nll,
                live_feedback_nll=live_nll,
                standard_k1_nll=standard_k1_nll,
                live_feedback_minus_exact_k_pass=live_nll - exact_nll,
                live_feedback_minus_standard_k1=live_nll - standard_k1_nll,
                retained_k4_improvement=retained,
                exact_to_live_kl=(
                    sum(kl_sum_by_offset[:horizon]) / distribution_count
                ),
                top1_agreement=(
                    sum(top1_equal_by_offset[:horizon]) / distribution_count
                ),
                hidden_delta_rms=math.sqrt(
                    sum(delta_sq_by_step[:horizon]) / hidden_count
                ),
                hidden_cosine=(
                    sum(cosine_sum_by_step[:horizon]) / cosine_count
                ),
            )
        )

    by_mode = {}
    for name, offsets in scores.items():
        total = NLLAccumulator()
        for offset in offsets:
            total.merge(offset)
        by_mode[name] = total

    return FeedbackEvaluationResult(
        prefill_passes=prefill_passes,
        blocks=limit,
        prompt_tokens=prompt_tokens,
        continuation_tokens=continuation_tokens,
        predicted_tokens_per_mode=sum(prediction_count_by_offset),
        horizons=tuple(horizon_results),
        exact_k_pass_nll_by_offset=exact_offset,
        live_feedback_nll_by_offset=live_offset,
        standard_k1_nll_by_offset=standard_k1_offset,
        exact_to_live_kl_by_offset=kl_offset,
        top1_agreement_by_offset=top1_offset,
        hidden_delta_rms_by_step=hidden_rms,
        hidden_cosine_by_step=hidden_cosine,
        predicted_tokens_by_offset=tuple(prediction_count_by_offset),
        nll_by_source_by_mode={
            name: total.by_source for name, total in by_mode.items()
        },
        predicted_tokens_by_source=dict(
            sorted(by_mode["exact_k_pass"].source_tokens.items())
        ),
        evaluation=packed_evaluation_metadata(
            model,
            dataset,
            device=device,
            autocast_dtype=autocast_dtype,
            blocks=limit,
            policy={
                "modes": ["exact_k_pass", "live_feedback", "standard_k1"],
                "prefill_passes": prefill_passes,
                "standard_prefill_passes": 1,
                "prompt_tokens": prompt_tokens,
                "continuation_tokens": continuation_tokens,
                "initialization": "data_prefix",
                "teacher_forced": True,
                "generation": False,
            },
            target_coverage=(
                "linguistic_tokens_in_continuation; prompt_unscored; no_added_bos"
            ),
        ),
    )
