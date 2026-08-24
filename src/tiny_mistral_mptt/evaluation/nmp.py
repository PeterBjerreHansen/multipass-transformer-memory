from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch

from ..data.packed_dataset import PackedTokenDataset
from ..variants.multipass import MultiPassVariant


@dataclass(frozen=True)
class NMPEvaluationResult:
    passes: int
    blocks: int
    predicted_tokens: int
    metrics: dict[str, float]


def _is_rms_metric(name: str) -> bool:
    return name.endswith("_rms")


def _is_count_metric(name: str) -> bool:
    return name.endswith(("_valid_queries", "_valid_events", "_valid_examples"))


def _metric_weight(name: str, metrics: dict[str, float], predicted_tokens: int) -> float:
    if name.startswith("pass_") or name == "ntp_loss":
        return float(predicted_tokens)
    if name.startswith("recurrent_nmp_"):
        return float(metrics.get("recurrent_nmp_valid_queries", 0.0))
    if name.startswith("bank_nmp_"):
        return float(metrics.get("bank_nmp_valid_examples", 0.0))
    return 1.0


@torch.no_grad()
def evaluate_nmp(
    model: MultiPassVariant,
    dataset: PackedTokenDataset,
    *,
    device: torch.device | str,
    passes: int,
    recurrent_nmp_loss_weights: Sequence[float] | None = None,
    bank_nmp_loss_weights: Sequence[float] | None = None,
    max_blocks: int | None = None,
) -> NMPEvaluationResult:
    """Evaluate online NMP deterministically at one fixed pass depth.

    The targets remain the model's configured detached online targets. This is
    therefore a generalization diagnostic on held-out sequences, not a claim
    that the teacher representation is stationary.
    """

    if passes < 1:
        raise ValueError("passes must be positive")
    if len(dataset) == 0:
        raise ValueError("validation dataset is empty")
    limit = len(dataset) if max_blocks is None else min(len(dataset), int(max_blocks))
    if limit <= 0:
        raise ValueError("max_blocks leaves no validation blocks")
    if model.recurrent_nmp_predictor is None and model.bank_nmp_predictor is None:
        raise ValueError("NMP evaluation requires at least one configured predictor")

    was_training = model.training
    model.eval()
    weighted_sums: dict[str, float] = defaultdict(float)
    metric_weights: dict[str, float] = defaultdict(float)
    count_sums: dict[str, float] = defaultdict(float)
    predicted_tokens = 0
    try:
        for index in range(limit):
            ids = dataset.batch([index], device=device)
            labels = model.build_lm_labels(ids)
            token_count = int(labels.ne(-100).sum().item())
            predicted_tokens += token_count
            output = model.compute_loss(
                ids,
                phase="B",
                passes=passes,
                recurrent_nmp_loss_weights=recurrent_nmp_loss_weights,
                bank_nmp_loss_weights=bank_nmp_loss_weights,
                nmp_weight_scale=1.0,
            )
            for name, raw_value in output.metrics.items():
                value = float(raw_value)
                if _is_count_metric(name):
                    count_sums[name] += value
                    continue
                weight = _metric_weight(name, output.metrics, token_count)
                if weight <= 0:
                    continue
                weighted_sums[name] += (
                    value * value if _is_rms_metric(name) else value
                ) * weight
                metric_weights[name] += weight
    finally:
        model.train(was_training)

    metrics = {
        name: (
            math.sqrt(max(weighted_sums[name] / weight, 0.0))
            if _is_rms_metric(name)
            else weighted_sums[name] / weight
        )
        for name, weight in sorted(metric_weights.items())
    }
    metrics.update(count_sums)
    return NMPEvaluationResult(
        passes=passes,
        blocks=limit,
        predicted_tokens=predicted_tokens,
        metrics=metrics,
    )
