"""Pass-indexed feedback interventions through the multipass model seam."""
from __future__ import annotations

from collections import defaultdict
import math

import torch

from ..feedback import HybridPassSource
from ..variants.multipass import HiddenRun, MultiPassVariant
from .common import (
    NLLAccumulator,
    block_limit,
    evaluation_context,
    packed_evaluation_metadata,
    perplexity,
    source_name,
)


def _zeros_like(source: torch.Tensor | HybridPassSource):
    if isinstance(source, HybridPassSource):
        return HybridPassSource(
            torch.zeros_like(source.recurrent_hidden),
            torch.zeros_like(source.memory_attention_hidden),
        )
    return torch.zeros_like(source)


def _hybrid_sources(
    real: HybridPassSource,
    mismatch: HybridPassSource,
) -> dict[str, HybridPassSource]:
    zero_recurrent = torch.zeros_like(real.recurrent_hidden)
    zero_attention = torch.zeros_like(real.memory_attention_hidden)
    return {
        "zero_recurrent_real_attention": HybridPassSource(
            zero_recurrent, real.memory_attention_hidden
        ),
        "mismatched_recurrent_real_attention": HybridPassSource(
            mismatch.recurrent_hidden, real.memory_attention_hidden
        ),
        "real_recurrent_zero_attention": HybridPassSource(
            real.recurrent_hidden, zero_attention
        ),
        "real_recurrent_mismatched_attention": HybridPassSource(
            real.recurrent_hidden, mismatch.memory_attention_hidden
        ),
    }


def _real_runs(
    model: MultiPassVariant,
    input_ids: torch.Tensor,
    *,
    passes: int,
) -> tuple[HiddenRun, ...]:
    runs = [model._run_first_state(input_ids)]
    for _ in range(1, passes):
        runs.append(
            model.run_feedback_transition(input_ids, runs[-1].feedback_source)
        )
    return tuple(runs)


def evaluate_memory_interventions(
    model,
    dataset,
    *,
    device,
    passes: int = 4,
    transitions: tuple[int, ...] | list[int] | None = None,
    max_blocks: int | None = None,
    autocast_dtype: str | None = None,
) -> dict:
    """Compare real, zero, mismatched, and true-bypass transitions.

    A transition numbered ``p`` supplies the source produced by pass ``p-1``
    to pass ``p``. Mismatch donors are independently processed to that same
    depth. True bypass does not call a reader or merger at all.
    """
    if not isinstance(model, MultiPassVariant):
        raise ValueError("loaded model does not support feedback interventions")
    if passes < 2:
        raise ValueError("interventions require passes >= 2")
    selected = (
        tuple(range(2, passes + 1))
        if transitions is None
        else tuple(sorted({int(value) for value in transitions}))
    )
    if not selected or selected[0] < 2 or selected[-1] > passes:
        raise ValueError("transitions must lie in [2, passes]")
    blocks = block_limit(dataset, max_blocks)
    if len(dataset) < 2:
        raise ValueError(
            "memory interventions require at least two validation blocks for a mismatch"
        )

    baseline = NLLAccumulator()
    totals: dict[int, dict[str, NLLAccumulator]] = {
        transition: {} for transition in selected
    }
    delta_sums: dict[int, dict[str, float]] = {
        transition: defaultdict(float) for transition in selected
    }
    mismatch_pairs: list[dict[str, int | str]] = []

    with evaluation_context(model, device=device, autocast_dtype=autocast_dtype):
        for index in range(blocks):
            ids = dataset.batch([index], device=device)
            mismatch_ids = dataset.batch([(index + 1) % len(dataset)], device=device)
            labels = model.build_lm_labels(ids)
            source_name_value = source_name(dataset, index)
            donor_index = (index + 1) % len(dataset)
            mismatch_pairs.append(
                {
                    "target_block_index": index,
                    "target_source": source_name_value,
                    "donor_block_index": donor_index,
                    "donor_source": source_name(dataset, donor_index),
                }
            )
            real_runs = _real_runs(model, ids, passes=passes)
            mismatch_runs = _real_runs(model, mismatch_ids, passes=passes)
            baseline.add(
                model.backbone.lm_head(real_runs[0].hidden_states),
                labels,
                source=source_name_value,
            )

            for transition in selected:
                source_index = transition - 2
                real_source = real_runs[source_index].feedback_source
                mismatch_source = mismatch_runs[source_index].feedback_source
                condition_sources: dict[
                    str, torch.Tensor | HybridPassSource | None
                ] = {
                    "real_memory": real_source,
                    "zero_memory": _zeros_like(real_source),
                    "mismatched_memory": mismatch_source,
                    "true_bypass": None,
                }
                if isinstance(real_source, HybridPassSource):
                    if not isinstance(mismatch_source, HybridPassSource):
                        raise TypeError("hybrid mismatch source has the wrong type")
                    condition_sources.update(
                        _hybrid_sources(real_source, mismatch_source)
                    )

                condition_hiddens: dict[str, torch.Tensor] = {}
                for name, condition_source in condition_sources.items():
                    if condition_source is None:
                        run = model.run_feedback_transition(
                            ids, real_source, bypass=True
                        )
                    else:
                        run = model.run_feedback_transition(ids, condition_source)
                    condition_hiddens[name] = run.hidden_states

                real_hidden = condition_hiddens["real_memory"]
                for name, hidden in condition_hiddens.items():
                    total = totals[transition].setdefault(name, NLLAccumulator())
                    total.add(
                        model.backbone.lm_head(hidden),
                        labels,
                        source=source_name_value,
                    )
                    reference = (
                        condition_hiddens["true_bypass"]
                        if name == "real_memory"
                        else real_hidden
                    )
                    delta_sums[transition][name] += float(
                        (hidden.float() - reference.float()).square().mean().cpu()
                    )

    def summarize(total: NLLAccumulator) -> dict:
        return {
            "nll": total.mean,
            "perplexity": perplexity(total.mean),
            "predicted_tokens": total.tokens,
            "nll_by_source": total.by_source,
            "predicted_tokens_by_source": dict(sorted(total.source_tokens.items())),
        }

    transition_results: dict[str, dict] = {}
    for transition in selected:
        conditions = {}
        for name, total in totals[transition].items():
            conditions[name] = {
                **summarize(total),
                "hidden_delta_rms": math.sqrt(
                    delta_sums[transition][name] / blocks
                ),
                "hidden_delta_reference": (
                    "true_bypass" if name == "real_memory" else "real_memory"
                ),
            }
        transition_results[str(transition)] = {
            "source_pass": transition - 1,
            "read_pass": transition,
            "mismatch_donor_depth": transition - 1,
            "conditions": conditions,
        }

    return {
        "variant": model.variant_name,
        "blocks": blocks,
        "passes": passes,
        "intervention_scope": "configurable_feedback_transitions",
        "baseline_pass1": summarize(baseline),
        "transitions": transition_results,
        "evaluation": packed_evaluation_metadata(
            model,
            dataset,
            device=device,
            autocast_dtype=autocast_dtype,
            blocks=blocks,
            policy={
                "forward": "exact_k_pass",
                "transitions": list(selected),
                "interventions": list(totals[selected[0]]),
                "true_bypass": "feedback_path_not_executed",
                "teacher_forced": True,
                "generation": False,
                "mismatch_donor": "(scored_block_index + 1) % available_blocks",
                "mismatch_donor_processed_to_source_pass": True,
                "mismatch_pairs": mismatch_pairs,
            },
        ),
    }
