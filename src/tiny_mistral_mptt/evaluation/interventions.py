"""Simple single-transition interventions; configurable-depth ablations are deferred."""
from __future__ import annotations

from collections import defaultdict
import math

import torch

from ..feedback import HybridPassSource
from ..variants.memory_add import MemoryAddVariant
from ..variants.recirculation import RecirculationVariant
from ..variants.recurrent_memory import RecurrentMemoryVariant
from ..variants.memory_attention_recurrent_hybrid import (
    MemoryAttentionRecurrentHybridVariant,
)
from ..variants.memory_attention import MemoryAttentionVariant

from .common import (
    NLLAccumulator, block_limit, evaluation_context, packed_evaluation_metadata,
    perplexity, source_name,
)


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().detach().cpu())


def _condition_hiddens(
    model,
    ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    real: torch.Tensor | HybridPassSource,
    mismatch: torch.Tensor | HybridPassSource,
) -> dict[str, torch.Tensor]:
    if isinstance(model, MemoryAttentionRecurrentHybridVariant):
        if not isinstance(real, HybridPassSource) or not isinstance(
            mismatch, HybridPassSource
        ):
            raise TypeError("hybrid intervention requires HybridPassSource")
        zero_recurrent = torch.zeros_like(real.recurrent_hidden)
        zero_memory = torch.zeros_like(real.memory_attention_hidden)

        def run(recurrent_hidden, memory_attention_hidden):
            source = HybridPassSource(recurrent_hidden, memory_attention_hidden)
            return model._run_feedback_state(
                ids, token_embeddings, source
            ).hidden_states

        return {
            "real_memory": run(real.recurrent_hidden, real.memory_attention_hidden),
            "zero_memory": run(zero_recurrent, zero_memory),
            "mismatched_memory": run(
                mismatch.recurrent_hidden, mismatch.memory_attention_hidden
            ),
            "zero_recurrent_real_attention": run(
                zero_recurrent, real.memory_attention_hidden
            ),
            "mismatched_recurrent_real_attention": run(
                mismatch.recurrent_hidden, real.memory_attention_hidden
            ),
            "real_recurrent_zero_attention": run(
                real.recurrent_hidden, zero_memory
            ),
            "real_recurrent_mismatched_attention": run(
                real.recurrent_hidden, mismatch.memory_attention_hidden
            ),
        }
    if not isinstance(real, torch.Tensor) or not isinstance(mismatch, torch.Tensor):
        raise TypeError("non-hybrid intervention requires tensor feedback sources")
    zero = torch.zeros_like(real)
    return {
        "real_memory": model._run_feedback_hidden(ids, token_embeddings, real),
        "zero_memory": model._run_feedback_hidden(ids, token_embeddings, zero),
        "mismatched_memory": model._run_feedback_hidden(ids, token_embeddings, mismatch),
    }


def evaluate_memory_interventions(
    model, dataset, *, device, max_blocks: int | None = None,
    autocast_dtype: str | None = None,
) -> dict:
    if not isinstance(model, (
        MemoryAddVariant, RecirculationVariant, RecurrentMemoryVariant,
        MemoryAttentionVariant,
    )):
        raise ValueError("loaded model does not support memory interventions")
    blocks = block_limit(dataset, max_blocks)
    if len(dataset) < 2:
        raise ValueError("memory interventions require at least two validation blocks for a mismatch")
    baseline = NLLAccumulator()
    totals: dict[str, NLLAccumulator] = {}
    delta_sums = defaultdict(float)
    embedding_rms_sum = residual_rms_sum = 0.0
    with evaluation_context(model, device=device, autocast_dtype=autocast_dtype):
        for index in range(blocks):
            ids = dataset.batch([index], device=device)
            mismatch_ids = dataset.batch([(index + 1) % len(dataset)], device=device)
            labels = model.build_lm_labels(ids)
            source = source_name(dataset, index)
            token_embeddings = model.input_embeddings(ids)
            first_run = model._run_first_state(ids)
            mismatch_run = model._run_first_state(mismatch_ids)
            first_hidden = first_run.hidden_states
            baseline.add(model.backbone.lm_head(first_hidden), labels, source=source)
            conditions = _condition_hiddens(
                model, ids, token_embeddings, first_run.feedback_source, mismatch_run.feedback_source,
            )
            for name, hidden in conditions.items():
                total = totals.setdefault(name, NLLAccumulator())
                total.add(model.backbone.lm_head(hidden), labels, source=source)
                delta_sums[name] += float(
                    (hidden.float() - first_hidden.float()).square().mean().cpu()
                )
            if isinstance(model, MemoryAddVariant):
                embedding_rms_sum += _rms(token_embeddings[:, 1:, :])
                residual = model.memory_residual(first_hidden)
                residual_rms_sum += _rms(residual[:, 1:, :])

    def summarize(total):
        return {
            "nll": total.mean, "perplexity": perplexity(total.mean),
            "predicted_tokens": total.tokens, "nll_by_source": total.by_source,
            "predicted_tokens_by_source": dict(sorted(total.source_tokens.items())),
        }

    result = {
        "variant": model.variant_name,
        "blocks": blocks,
        "intervention_scope": "single_feedback_transition",
        "baseline_pass1": summarize(baseline),
        "evaluation": packed_evaluation_metadata(
            model, dataset, device=device, autocast_dtype=autocast_dtype, blocks=blocks,
            policy={
                "forward": "parallel_multipass", "source_pass": 1, "read_pass": 2,
                "interventions": list(totals), "teacher_forced": True, "generation": False,
                "mismatch_donor": "(scored_block_index + 1) % available_blocks",
            },
        ),
    }
    for name, total in totals.items():
        result[name] = {
            **summarize(total),
            "hidden_delta_rms": math.sqrt(delta_sums[name] / blocks),
        }
    if isinstance(model, MemoryAddVariant):
        embedding_rms = embedding_rms_sum / blocks
        residual_rms = residual_rms_sum / blocks
        result["memory_add_scales"] = {
            "embedding_rms_noninitial": embedding_rms,
            "memory_residual_rms_noninitial": residual_rms,
            "residual_to_embedding_rms_ratio": (
                residual_rms / embedding_rms if embedding_rms > 0 else float("nan")
            ),
        }
    return result
