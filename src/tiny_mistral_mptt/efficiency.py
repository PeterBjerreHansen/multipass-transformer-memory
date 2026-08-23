from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class BatchQualification:
    selected_microbatch: int
    sequence_length: int
    grad_accum_steps: int
    optimizer_batch_tokens: int
    reference_optimizer_batch_tokens: int
    changes_optimizer_batch: bool
    local_grad_accum_steps_to_match: int | None
    efficiency_fraction: float
    throughput_fraction_by_variant: dict[str, float]
    common_successful_microbatches: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["common_successful_microbatches"] = list(self.common_successful_microbatches)
        return payload


def recommend_cuda_microbatch(
    benchmark_document: dict[str, Any],
    *,
    variants: Iterable[str] = ("recirculation", "bank"),
    passes: int = 2,
    sequence_length: int = 2048,
    autocast_dtype: str | None = "bfloat16",
    efficiency_fraction: float = 0.90,
    reference_optimizer_batch_tokens: int = 2048,
) -> BatchQualification:
    """Choose the smallest common efficient CUDA microbatch.

    The CUDA qualification intentionally measures ``grad_accum_steps=1``. The
    selector therefore treats microbatch size as the only hardware axis and
    reports when the recommended microbatch would also change the scientific
    optimizer-batch size relative to the reference trajectory.
    """
    if not 0.0 < efficiency_fraction <= 1.0:
        raise ValueError("efficiency_fraction must lie in (0, 1]")
    if passes <= 0 or sequence_length <= 0 or reference_optimizer_batch_tokens <= 0:
        raise ValueError("passes, sequence_length, and reference batch must be positive")

    variant_names = tuple(dict.fromkeys(str(value) for value in variants))
    if not variant_names:
        raise ValueError("at least one variant is required")

    rows = benchmark_document.get("results")
    if not isinstance(rows, list):
        raise ValueError("benchmark document must contain a results list")

    throughput: dict[str, dict[int, float]] = {name: {} for name in variant_names}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "ok":
            continue
        variant = str(row.get("variant", ""))
        if variant not in throughput:
            continue
        if int(row.get("passes", -1)) != passes:
            continue
        if int(row.get("sequence_length", -1)) != sequence_length:
            continue
        if int(row.get("grad_accum_steps", 1)) != 1:
            continue
        if row.get("autocast_dtype") != autocast_dtype:
            continue
        if not str(row.get("device", "")).startswith("cuda"):
            continue
        batch = int(row.get("batch_size", 0))
        rate = float(row.get("unique_tokens_per_second", 0.0))
        if batch > 0 and rate > 0.0:
            throughput[variant][batch] = rate

    missing = [name for name, values in throughput.items() if not values]
    if missing:
        raise ValueError(f"no successful matching CUDA rows for variants: {missing}")

    common = set.intersection(*(set(values) for values in throughput.values()))
    if not common:
        raise ValueError("no common successful microbatch exists across requested variants")

    common_batches = tuple(sorted(common))
    maxima = {
        name: max(values.values())
        for name, values in throughput.items()
    }
    eligible: list[int] = []
    for batch in common_batches:
        if all(
            throughput[name][batch] >= efficiency_fraction * maxima[name]
            for name in variant_names
        ):
            eligible.append(batch)
    if not eligible:
        raise ValueError("no common microbatch reaches the requested efficiency fraction")

    selected = min(eligible)
    optimizer_batch_tokens = selected * sequence_length
    local_accum = (
        reference_optimizer_batch_tokens // optimizer_batch_tokens
        if reference_optimizer_batch_tokens % optimizer_batch_tokens == 0
        else None
    )
    fractions = {
        name: throughput[name][selected] / maxima[name]
        for name in variant_names
    }
    return BatchQualification(
        selected_microbatch=selected,
        sequence_length=sequence_length,
        grad_accum_steps=1,
        optimizer_batch_tokens=optimizer_batch_tokens,
        reference_optimizer_batch_tokens=reference_optimizer_batch_tokens,
        changes_optimizer_batch=optimizer_batch_tokens != reference_optimizer_batch_tokens,
        local_grad_accum_steps_to_match=local_accum,
        efficiency_fraction=efficiency_fraction,
        throughput_fraction_by_variant=fractions,
        common_successful_microbatches=common_batches,
    )
