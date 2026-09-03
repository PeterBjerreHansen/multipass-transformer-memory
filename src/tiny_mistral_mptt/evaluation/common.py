"""Shared execution, target scoring and packed-subset identity for evaluation.

Callers choose computation, not how to average losses. Labels remain owned by
the model: this preserves linguistic targets across input-only control slots.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from ..data.manifest import file_sha256
from ..precision import autocast_context


@contextmanager
def evaluation_context(model, *, device, autocast_dtype: str | None = None):
    """Use explicit compute precision and restore training mode even on failure."""
    was_training = model.training
    model.eval()
    try:
        precision = (
            autocast_context(device, autocast_dtype) if autocast_dtype is not None
            else torch.autocast(device_type=torch.device(device).type, enabled=False)
        )
        with torch.no_grad(), precision:
            yield
    finally:
        model.train(was_training)


def score_targets(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    """Sum FP32 cross-entropy over aligned targets; -100 marks unscored slots."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits and labels must align at every prediction position")
    count = int(labels.ne(-100).sum().item())
    if not count:
        return 0.0, 0
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.to(logits.device).reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return float(loss.detach().cpu()), count


def perplexity(nll: float) -> float:
    return math.exp(min(nll, 50.0))


@dataclass
class NLLAccumulator:
    loss: float = 0.0
    tokens: int = 0
    source_loss: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    source_tokens: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, logits, labels, *, source: str | None = None) -> None:
        value, count = score_targets(logits, labels)
        self.loss += value
        self.tokens += count
        if source is not None:
            self.source_loss[source] += value
            self.source_tokens[source] += count

    def merge(self, other: NLLAccumulator) -> None:
        self.loss += other.loss
        self.tokens += other.tokens
        for source in other.source_tokens:
            self.source_loss[source] += other.source_loss[source]
            self.source_tokens[source] += other.source_tokens[source]

    @property
    def mean(self) -> float:
        if not self.tokens:
            raise ValueError("selected evaluation contains no linguistic prediction targets")
        return self.loss / self.tokens

    @property
    def by_source(self) -> dict[str, float]:
        return {
            source: self.source_loss[source] / count
            for source, count in sorted(self.source_tokens.items()) if count
        }


def block_limit(dataset, max_blocks: int | None) -> int:
    if len(dataset) == 0:
        raise ValueError("validation dataset is empty")
    if max_blocks is not None and max_blocks <= 0:
        raise ValueError("max_blocks must be positive when provided")
    return len(dataset) if max_blocks is None else min(len(dataset), max_blocks)


def source_name(dataset, index: int) -> str:
    source_id = dataset.source_id(index)
    return next(name for name, value in dataset.manifest.source_ids.items() if value == source_id)


def precision_metadata(model, *, device, autocast_dtype: str | None) -> dict:
    return {
        "device": str(torch.device(device)),
        "parameter_dtypes": sorted({str(p.dtype).removeprefix("torch.") for p in model.parameters()}),
        "autocast_dtype": autocast_dtype,
        "loss_dtype": "float32",
    }


def packed_evaluation_metadata(
    model, dataset, *, device, autocast_dtype, blocks: int, policy: dict,
    target_coverage: str = "next_linguistic_token_within_block; first_token_unscored; no_added_bos",
) -> dict:
    """Identify the actual prefix subset, including architecture-inserted slots.

    Artifact hashes are the manifest's declared hashes (not a fresh multi-GB
    integrity scan on each routine check). Artifact verification is separate.
    Lightweight in-memory diagnostic datasets may have no manifest.
    """
    data = {
        "split": getattr(dataset, "split", None),
        "selection": {"kind": "prefix_blocks", "start": 0, "stop": blocks},
        "available_blocks": len(dataset),
        "physical_sequence_length": dataset.sequence_length,
        "linguistic_sequence_length": getattr(dataset, "linguistic_sequence_length", dataset.sequence_length),
        "memory_token_interval": getattr(dataset, "interval", None),
    }
    artifact_dir = getattr(dataset, "artifact_dir", None)
    if artifact_dir is not None:
        data["directory"] = str(artifact_dir.resolve())
        data["manifest_sha256"] = file_sha256(artifact_dir / "manifest.json")
        info = getattr(dataset.manifest, dataset.split)
        data["declared_token_sha256"] = info.data_sha256
        data["declared_source_sha256"] = info.source_sha256
    return {
        "schema_version": 2,
        "precision": precision_metadata(model, device=device, autocast_dtype=autocast_dtype),
        "policy": policy,
        "data": data,
        "target_coverage": target_coverage,
        "aggregation": "summed_token_nll / scored_linguistic_tokens",
    }
