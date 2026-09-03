from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..data.packed_dataset import PackedTokenDataset
from ..variants.multipass import MultiPassVariant
from .common import (
    NLLAccumulator, block_limit, evaluation_context, packed_evaluation_metadata,
    perplexity, source_name,
)


@dataclass(frozen=True)
class PassDepthResult:
    passes: int
    blocks: int
    predicted_tokens: int
    nll_by_pass: tuple[float, ...]
    perplexity_by_pass: tuple[float, ...]
    hidden_delta_rms: tuple[float, ...]
    nll_by_source_by_pass: tuple[dict[str, float], ...]
    predicted_tokens_by_source: dict[str, int]
    evaluation: dict

    @property
    def final_nll(self) -> float:
        return self.nll_by_pass[-1]

    @property
    def final_perplexity(self) -> float:
        return self.perplexity_by_pass[-1]


def evaluate_pass_depth(
    model,
    dataset: PackedTokenDataset,
    *,
    device: torch.device | str,
    passes: int,
    max_blocks: int | None = None,
    autocast_dtype: str | None = None,
) -> PassDepthResult:
    """Shared parallel evaluator for trainer validation and standalone NLL.

    K=1 also accepts ordinary single-pass models. Blocks are deliberately still
    serialized; max_blocks is a subset limit, not an inference batch size.
    """
    if passes < 1:
        raise ValueError("passes must be positive")
    if not isinstance(model, MultiPassVariant) and passes != 1:
        raise ValueError("single-pass variants support only passes=1")
    limit = block_limit(dataset, max_blocks)
    totals = [NLLAccumulator() for _ in range(passes)]
    delta_sq_sums = [0.0] * max(passes - 1, 0)
    delta_counts = [0] * max(passes - 1, 0)
    with evaluation_context(model, device=device, autocast_dtype=autocast_dtype):
        for index in range(limit):
            ids = dataset.batch([index], device=device)
            labels = model.build_lm_labels(ids)
            source = source_name(dataset, index)
            if isinstance(model, MultiPassVariant):
                outputs = model.compute_passes(ids, passes=passes, phase="B").passes
            else:
                outputs = (model(ids, use_cache=False),)
            for total, output in zip(totals, outputs, strict=True):
                total.add(output.logits, labels, source=source)
            for transition in range(1, passes):
                # Preserve the previous diagnostic: subtraction in compute
                # dtype, then FP32 reduction over all physical hidden slots.
                delta = outputs[transition].hidden_states - outputs[transition - 1].hidden_states
                delta_sq_sums[transition - 1] += float(delta.float().square().sum().cpu())
                delta_counts[transition - 1] += delta.numel()

    nll = tuple(total.mean for total in totals)
    return PassDepthResult(
        passes=passes,
        blocks=limit,
        predicted_tokens=totals[0].tokens,
        nll_by_pass=nll,
        perplexity_by_pass=tuple(perplexity(value) for value in nll),
        hidden_delta_rms=tuple(
            math.sqrt(total / count) for total, count in zip(delta_sq_sums, delta_counts, strict=True)
        ),
        nll_by_source_by_pass=tuple(total.by_source for total in totals),
        predicted_tokens_by_source=dict(sorted(totals[0].source_tokens.items())),
        evaluation=packed_evaluation_metadata(
            model, dataset, device=device, autocast_dtype=autocast_dtype, blocks=limit,
            policy={"forward": "parallel_multipass", "passes": passes,
                    "teacher_forced": True, "generation": False},
        ),
    )
