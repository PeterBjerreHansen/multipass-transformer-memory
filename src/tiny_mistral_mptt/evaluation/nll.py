from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from ..data.packed_dataset import PackedTokenDataset
from ..variants.multipass import MultiPassVariant


@dataclass(frozen=True)
class NLLResult:
    nll: float
    perplexity: float
    predicted_tokens: int
    blocks: int
    passes: int
    nll_by_source: dict[str, float]


@torch.no_grad()
def evaluate_nll(
    model,
    dataset: PackedTokenDataset,
    *,
    device: torch.device | str,
    passes: int = 1,
    max_blocks: int | None = None,
) -> NLLResult:
    if passes < 1:
        raise ValueError("passes must be positive")
    if len(dataset) == 0:
        raise ValueError("validation dataset is empty")
    if max_blocks is not None and max_blocks <= 0:
        raise ValueError("max_blocks must be positive when provided")
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    source_nll: dict[int, float] = defaultdict(float)
    source_tokens: dict[int, int] = defaultdict(int)
    limit = len(dataset) if max_blocks is None else min(len(dataset), max_blocks)
    try:
        for index in range(limit):
            ids = dataset.batch([index], device=device)
            if isinstance(model, MultiPassVariant):
                logits = model.compute_passes(ids, passes=passes, phase="B").final.logits.float()
            else:
                if passes != 1:
                    raise ValueError("single-pass variants support only passes=1")
                output = model(ids, use_cache=False)
                logits = output.logits.float()
            labels = model.build_lm_labels(ids)
            valid = labels.ne(-100)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.to(logits.device).reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            value = float(losses.detach().cpu())
            count = int(valid.sum().item())
            total_nll += value
            total_tokens += count
            source_id = dataset.source_id(index)
            source_nll[source_id] += value
            source_tokens[source_id] += count
    finally:
        model.train(was_training)
    id_to_name = {value: key for key, value in dataset.manifest.source_ids.items()}
    by_source = {
        id_to_name[source_id]: source_nll[source_id] / source_tokens[source_id]
        for source_id in sorted(source_nll)
    }
    mean = total_nll / total_tokens
    return NLLResult(
        nll=mean,
        perplexity=math.exp(min(mean, 50.0)),
        predicted_tokens=total_tokens,
        blocks=limit,
        passes=passes,
        nll_by_source=by_source,
    )
