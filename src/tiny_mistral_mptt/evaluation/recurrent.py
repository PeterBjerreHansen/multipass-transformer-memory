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


@dataclass(frozen=True)
class RecurrentHorizonResult:
    horizon: int
    exact_nll: float
    recurrent_nll: float
    vanilla_nll: float
    recurrent_minus_exact: float
    recurrent_minus_vanilla: float
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
    vanilla_nll_by_offset: tuple[float, ...]
    hidden_delta_rms_by_step: tuple[float, ...]
    hidden_cosine_by_step: tuple[float, ...]


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


def _token_nll(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[float, int]:
    if logits.ndim != 2 or target.ndim != 2 or target.shape[1] != 1:
        raise ValueError("logits must be [B,V] and target [B,1]")
    if logits.shape[0] != target.shape[0]:
        raise ValueError("logits and target batch sizes differ")
    if valid is None:
        valid = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
    if valid.shape != (target.shape[0],):
        raise ValueError("valid must be bool [B]")
    if not bool(valid.any()):
        return 0.0, 0
    loss = F.cross_entropy(logits[valid].float(), target[valid, 0], reduction="sum")
    return float(loss.detach().cpu()), int(valid.sum().item())


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
) -> RecurrentEvaluationResult:
    """Compare exact K-stream and collapsed recurrent teacher-forced decoding.

    A pass-1 cached stream from the same checkpoint is evaluated alongside the
    two memory modes as the no-memory control.  Exact and recurrent states share
    one K-pass prefill, so their initial logits and the first processed-token
    transition are identical by construction; later differences measure the
    recurrent train/inference shift directly.
    """
    if prefill_passes < 1:
        raise ValueError("prefill_passes must be positive")
    if prompt_tokens < 1 or continuation_tokens < 1:
        raise ValueError("prompt_tokens and continuation_tokens must be positive")
    if prompt_tokens + continuation_tokens > dataset.sequence_length:
        raise ValueError("prompt + continuation exceeds packed sequence length")
    if len(dataset) == 0:
        raise ValueError("validation dataset is empty")
    limit = len(dataset) if max_blocks is None else min(len(dataset), int(max_blocks))
    if limit <= 0:
        raise ValueError("max_blocks leaves no validation blocks")
    report_horizons = _normalize_horizons(horizons, continuation_tokens)

    exact_loss_by_offset = [0.0] * continuation_tokens
    recurrent_loss_by_offset = [0.0] * continuation_tokens
    vanilla_loss_by_offset = [0.0] * continuation_tokens
    prediction_count_by_offset = [0] * continuation_tokens
    delta_sq_by_step = [0.0] * continuation_tokens
    delta_count_by_step = [0] * continuation_tokens
    cosine_sum_by_step = [0.0] * continuation_tokens
    cosine_count_by_step = [0] * continuation_tokens

    was_training = model.training
    model.eval()
    try:
        for block_index in range(limit):
            ids = dataset.batch([block_index], device=device)
            prompt = ids[:, :prompt_tokens]
            continuation = ids[
                :, prompt_tokens : prompt_tokens + continuation_tokens
            ]

            exact = prefill_exact(model, prompt, passes=prefill_passes)
            recurrent = recurrent_from_exact(exact, decode_mode="feedback")
            vanilla = prefill_exact(model, prompt, passes=1)

            for offset in range(continuation_tokens):
                target = continuation[:, offset : offset + 1]
                valid_target = ~model.control_token_mask(target)[:, 0]
                exact_loss, count = _token_nll(exact.next_token_logits, target, valid_target)
                recurrent_loss, _ = _token_nll(recurrent.next_token_logits, target, valid_target)
                vanilla_loss, _ = _token_nll(vanilla.next_token_logits, target, valid_target)
                exact_loss_by_offset[offset] += exact_loss
                recurrent_loss_by_offset[offset] += recurrent_loss
                vanilla_loss_by_offset[offset] += vanilla_loss
                prediction_count_by_offset[offset] += count

                # Process the observed target even at the last offset so the
                # latent drift after every continuation token is measured.
                exact = exact_decode_step(model, exact, target)
                recurrent = recurrent_decode_step(model, recurrent, target)
                vanilla = exact_decode_step(model, vanilla, target)

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
    finally:
        model.train(was_training)

    def _offset_means(values: list[float]) -> tuple[float, ...]:
        return tuple(
            value / count if count else float("nan")
            for value, count in zip(values, prediction_count_by_offset, strict=True)
        )

    exact_offset = _offset_means(exact_loss_by_offset)
    recurrent_offset = _offset_means(recurrent_loss_by_offset)
    vanilla_offset = _offset_means(vanilla_loss_by_offset)
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
        exact_nll = sum(exact_loss_by_offset[:horizon]) / count
        recurrent_nll = sum(recurrent_loss_by_offset[:horizon]) / count
        vanilla_nll = sum(vanilla_loss_by_offset[:horizon]) / count
        horizon_results.append(
            RecurrentHorizonResult(
                horizon=horizon,
                exact_nll=exact_nll,
                recurrent_nll=recurrent_nll,
                vanilla_nll=vanilla_nll,
                recurrent_minus_exact=recurrent_nll - exact_nll,
                recurrent_minus_vanilla=recurrent_nll - vanilla_nll,
                hidden_delta_rms=hidden_rms[horizon - 1],
                hidden_cosine=hidden_cosine[horizon - 1],
            )
        )

    return RecurrentEvaluationResult(
        prefill_passes=prefill_passes,
        blocks=limit,
        prompt_tokens=prompt_tokens,
        continuation_tokens=continuation_tokens,
        predicted_tokens_per_mode=sum(prediction_count_by_offset),
        horizons=tuple(horizon_results),
        exact_nll_by_offset=exact_offset,
        recurrent_nll_by_offset=recurrent_offset,
        vanilla_nll_by_offset=vanilla_offset,
        hidden_delta_rms_by_step=hidden_rms,
        hidden_cosine_by_step=hidden_cosine,
    )
