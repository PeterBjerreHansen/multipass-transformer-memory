from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..data.packed_dataset import PackedTokenDataset
from ..variants.memory_attention_fusion import MemoryAttentionFusion
from .common import block_limit, evaluation_context, packed_evaluation_metadata


@dataclass
class _SignalMoments:
    square_sum: float = 0.0
    count: int = 0

    def add(self, values: torch.Tensor) -> None:
        values = values.detach().float()
        self.square_sum += float(values.square().sum().cpu())
        self.count += values.numel()

    @property
    def rms(self) -> float:
        return math.sqrt(self.square_sum / self.count) if self.count else 0.0


@dataclass
class _GateMoments:
    bins: int = 1000
    value_sum: float = 0.0
    square_sum: float = 0.0
    count: int = 0
    below_point_05: int = 0
    above_point_95: int = 0
    histogram: torch.Tensor | None = None

    def add(self, values: torch.Tensor) -> None:
        values = values.detach().float()
        self.value_sum += float(values.sum().cpu())
        self.square_sum += float(values.square().sum().cpu())
        self.count += values.numel()
        self.below_point_05 += int(values.lt(0.05).sum().cpu())
        self.above_point_95 += int(values.gt(0.95).sum().cpu())
        histogram = torch.histc(values, bins=self.bins, min=0.0, max=1.0).cpu()
        if self.histogram is None:
            self.histogram = histogram
        else:
            self.histogram += histogram

    def _quantile(self, probability: float) -> float:
        if not self.count or self.histogram is None:
            raise ValueError("cannot take a quantile of empty gate values")
        target = probability * (self.count - 1)
        index = int(
            torch.searchsorted(
                self.histogram.cumsum(0), torch.tensor(target + 1.0)
            ).item()
        )
        return (min(index, self.bins - 1) + 0.5) / self.bins

    def result(self) -> dict[str, float | int]:
        if not self.count:
            raise ValueError("cannot summarize empty gate values")
        mean = self.value_sum / self.count
        variance = max(self.square_sum / self.count - mean * mean, 0.0)
        return {
            "elements": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "p05": self._quantile(0.05),
            "p50": self._quantile(0.50),
            "p95": self._quantile(0.95),
            "fraction_below_0_05": self.below_point_05 / self.count,
            "fraction_above_0_95": self.above_point_95 / self.count,
        }


@dataclass
class _Cell:
    valid_tokens: int = 0
    attention: _SignalMoments | None = None
    destination: _SignalMoments | None = None
    attention_contribution: _SignalMoments | None = None
    destination_contribution: _SignalMoments | None = None
    alpha: _GateMoments | None = None
    beta: _GateMoments | None = None

    def __post_init__(self) -> None:
        self.attention = _SignalMoments()
        self.destination = _SignalMoments()
        self.attention_contribution = _SignalMoments()
        self.destination_contribution = _SignalMoments()

    def add(
        self,
        module: MemoryAttentionFusion,
        destination: torch.Tensor,
        attention: torch.Tensor,
        available: torch.Tensor,
    ) -> None:
        self.valid_tokens += int(available.sum().cpu())
        attention = attention[available]
        destination = destination[available]
        gates = module.gate_values(attention, destination)
        attention_contribution = (
            attention if gates.alpha is None else gates.alpha * attention
        )
        destination_contribution = (
            destination if gates.beta is None else gates.beta * destination
        )
        assert self.attention is not None
        assert self.destination is not None
        assert self.attention_contribution is not None
        assert self.destination_contribution is not None
        self.attention.add(attention)
        self.destination.add(destination)
        self.attention_contribution.add(attention_contribution)
        self.destination_contribution.add(destination_contribution)
        if gates.alpha is not None:
            if self.alpha is None:
                self.alpha = _GateMoments()
            self.alpha.add(gates.alpha)
        if gates.beta is not None:
            if self.beta is None:
                self.beta = _GateMoments()
            self.beta.add(gates.beta)

    def result(self, *, pass_index: int, layer_index: int) -> dict:
        assert self.attention is not None
        assert self.destination is not None
        assert self.attention_contribution is not None
        assert self.destination_contribution is not None
        return {
            "pass": pass_index,
            "layer": layer_index,
            "valid_memory_tokens": self.valid_tokens,
            "attention_rms": self.attention.rms,
            "destination_rms": self.destination.rms,
            "attention_to_destination_rms_ratio": (
                self.attention.rms / self.destination.rms
                if self.destination.rms
                else None
            ),
            "attention_contribution_rms": self.attention_contribution.rms,
            "destination_contribution_rms": self.destination_contribution.rms,
            "alpha": None if self.alpha is None else self.alpha.result(),
            "beta": None if self.beta is None else self.beta.result(),
        }


def evaluate_attention_fusion(
    model,
    dataset: PackedTokenDataset,
    *,
    device: torch.device | str,
    passes: int,
    max_blocks: int | None = None,
    autocast_dtype: str | None = None,
) -> dict:
    """Measure fusion inputs, contributions, and gates by pass and reader site."""
    if passes < 2:
        raise ValueError("attention-fusion diagnostics require at least two passes")
    fusions = getattr(model, "memory_attention_fusions", None)
    if not fusions:
        raise ValueError("model has no Memory Attention fusion sites")
    limit = block_limit(dataset, max_blocks)
    cells: dict[tuple[int, int], _Cell] = {}
    calls_by_layer: dict[int, int] = {}

    def hook(layer_index: int):
        def collect(module, args, kwargs, output):
            del output
            destination, attention = args
            available = kwargs.get("available")
            if available is None:
                raise RuntimeError("fusion diagnostic requires explicit availability")
            call_index = calls_by_layer.get(layer_index, 0)
            calls_by_layer[layer_index] = call_index + 1
            pass_index = call_index + 2
            cells.setdefault((pass_index, layer_index), _Cell()).add(
                module, destination, attention, available
            )

        return collect

    handles = [
        module.register_forward_hook(hook(int(layer)), with_kwargs=True)
        for layer, module in fusions.items()
    ]
    try:
        with evaluation_context(model, device=device, autocast_dtype=autocast_dtype):
            for index in range(limit):
                calls_by_layer.clear()
                ids = dataset.batch([index], device=device)
                model.compute_passes(ids, passes=passes, phase="B")
                expected_calls = passes - 1
                if any(
                    calls_by_layer.get(int(layer), 0) != expected_calls
                    for layer in fusions
                ):
                    raise RuntimeError(
                        "unexpected Memory Attention fusion call count during evaluation"
                    )
    finally:
        for handle in handles:
            handle.remove()

    rows = [
        cell.result(pass_index=pass_index, layer_index=layer_index)
        for (pass_index, layer_index), cell in sorted(cells.items())
    ]
    evaluation = packed_evaluation_metadata(
        model,
        dataset,
        device=device,
        autocast_dtype=autocast_dtype,
        blocks=limit,
        policy={
            "forward": "parallel_multipass",
            "passes": passes,
            "teacher_forced": True,
            "generation": False,
            "positions": "strict_past_memory_available_only",
        },
    )
    evaluation["target_coverage"] = (
        "physical positions with at least one strict-past memory record"
    )
    evaluation["aggregation"] = (
        "RMS over valid feature elements; exact gate moments and saturation; "
        "1000-bin gate quantiles"
    )
    return {
        "passes": passes,
        "blocks": limit,
        "fusion": model.memory_attention_fusion,
        "sites": rows,
        "evaluation": evaluation,
    }
