#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import canonical_variant_name, load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.evaluation.provenance import (
    add_checkpoint_arguments,
    evaluation_provenance,
    load_evaluation_weights,
    render_or_write_json,
    seed_evaluation,
)
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.feedback import HybridPassSource
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant
from tiny_mistral_mptt.variants.bank_add_hybrid import MemoryAttentionAddHybridVariant
from tiny_mistral_mptt.variants.bank_recirculation_hybrid import (
    RecirculationStridedMemoryAttentionVariant,
)
from tiny_mistral_mptt.variants.bank_recurrent_hybrid import (
    MemoryAttentionRecurrentHybridVariant,
)
from tiny_mistral_mptt.variants.bank import MemoryAttentionVariant


SUPPORTED = {
    "memory_add",
    "recirculation",
    "bank",
    "bank_multiscale",
    "bank_add_hybrid",
    "bank_recirculation_hybrid",
}


def _nll(model, logits: torch.Tensor, ids: torch.Tensor) -> tuple[float, int]:
    labels = model.build_lm_labels(ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.to(logits.device).reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    count = int(labels.ne(-100).sum().item())
    return float(loss.detach().cpu()), count


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
        zero_bank = torch.zeros_like(real.bank_hidden)

        def run(recurrent_hidden, bank_hidden):
            source = HybridPassSource(recurrent_hidden, bank_hidden)
            return model._run_feedback_state(
                ids, token_embeddings, source
            ).hidden_states

        recurrent_label = "fast" if isinstance(model, MemoryAttentionAddHybridVariant) else "recurrent"
        return {
            "real_memory": run(real.recurrent_hidden, real.bank_hidden),
            "zero_memory": run(zero_recurrent, zero_bank),
            "mismatched_memory": run(
                mismatch.recurrent_hidden, mismatch.bank_hidden
            ),
            f"zero_{recurrent_label}_real_bank": run(
                zero_recurrent, real.bank_hidden
            ),
            f"mismatched_{recurrent_label}_real_bank": run(
                mismatch.recurrent_hidden, real.bank_hidden
            ),
            f"real_{recurrent_label}_zero_bank": run(
                real.recurrent_hidden, zero_bank
            ),
            f"real_{recurrent_label}_mismatched_bank": run(
                real.recurrent_hidden, mismatch.bank_hidden
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run real/zero/mismatched recurrence interventions. The hybrid also "
            "reports fast and Memory Attention channel interventions independently."
        )
    )
    parser.add_argument("--config", required=True)
    add_checkpoint_arguments(parser)
    parser.add_argument("--evaluation-data-dir", default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seed_evaluation(args.seed)
    cfg = load_experiment_config(args.config)
    if canonical_variant_name(cfg.variant) not in SUPPORTED:
        raise SystemExit(
            "evaluate_memory_interventions requires a MemoryAdd/Recirculation/Memory Attention variant"
        )
    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    if not isinstance(
        model,
            (
                MemoryAddVariant,
                RecirculationVariant,
                MemoryAttentionVariant,
                MemoryAttentionAddHybridVariant,
                RecirculationStridedMemoryAttentionVariant,
            ),
    ):
        raise SystemExit("loaded model does not support memory interventions")

    weights = load_evaluation_weights(
        model=model,
        config=cfg,
        checkpoint=args.checkpoint,
        initialized_baseline=args.initialized_baseline,
    )
    model.eval()

    evaluation_data_dir = args.evaluation_data_dir or cfg.data_dir
    dataset = load_packed_dataset_for_experiment(
        evaluation_data_dir,
        "validation",
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
    )
    blocks = len(dataset) if args.max_blocks is None else min(len(dataset), args.max_blocks)
    if blocks <= 0:
        raise SystemExit("no validation blocks selected")
    if len(dataset) < 2:
        raise SystemExit(
            "memory interventions require at least two validation blocks for a mismatch"
        )

    totals: dict[str, dict[str, float | int]] = {}
    baseline_loss = 0.0
    baseline_count = 0
    embedding_rms_sum = 0.0
    residual_rms_sum = 0.0

    with torch.no_grad():
        for index in range(blocks):
            ids = dataset.batch([index], device=device)
            mismatch_ids = dataset.batch([(index + 1) % len(dataset)], device=device)
            token_embeddings = model.input_embeddings(ids)
            first_run = model._run_first_state(ids)
            mismatch_run = model._run_first_state(mismatch_ids)
            first_hidden = first_run.hidden_states
            feedback_source = first_run.feedback_source
            mismatch_source = mismatch_run.feedback_source

            baseline_logits = model.backbone.lm_head(first_hidden).float()
            loss, count = _nll(model, baseline_logits, ids)
            baseline_loss += loss
            baseline_count += count

            conditions = _condition_hiddens(
                model, ids, token_embeddings, feedback_source, mismatch_source
            )
            for name, hidden in conditions.items():
                values = totals.setdefault(
                    name, {"loss": 0.0, "count": 0, "delta_sq": 0.0}
                )
                logits = model.backbone.lm_head(hidden).float()
                loss, count = _nll(model, logits, ids)
                delta_sq = float(
                    (hidden.float() - first_hidden.float())
                    .square()
                    .mean()
                    .detach()
                    .cpu()
                )
                values["loss"] = float(values["loss"]) + loss
                values["count"] = int(values["count"]) + count
                values["delta_sq"] = float(values["delta_sq"]) + delta_sq

            if isinstance(model, (MemoryAddVariant, MemoryAttentionAddHybridVariant)):
                embedding_rms_sum += _rms(token_embeddings[:, 1:, :])
                residual_rms_sum += _rms(model.memory_residual(first_hidden, ids)[:, 1:, :] if isinstance(model, MemoryAttentionAddHybridVariant) else model.memory_residual(first_hidden)[:, 1:, :])

    result: dict[str, object] = {
        "variant": cfg.variant,
        "blocks": blocks,
        "intervention_scope": "single_feedback_transition",
        "baseline_pass1": {
            "nll": baseline_loss / baseline_count,
            "perplexity": math.exp(baseline_loss / baseline_count),
        },
    }
    for name, values in totals.items():
        count = int(values["count"])
        nll = float(values["loss"]) / count
        result[name] = {
            "nll": nll,
            "perplexity": math.exp(nll),
            "hidden_delta_rms": math.sqrt(float(values["delta_sq"]) / blocks),
        }

    if isinstance(model, (MemoryAddVariant, MemoryAttentionAddHybridVariant)):
        embedding_rms = embedding_rms_sum / blocks
        residual_rms = residual_rms_sum / blocks
        result["memory_add_scales"] = {
            "embedding_rms_noninitial": embedding_rms,
            "memory_residual_rms_noninitial": residual_rms,
            "residual_to_embedding_rms_ratio": (
                residual_rms / embedding_rms if embedding_rms > 0 else float("nan")
            ),
        }

    document = {
        "evaluation_kind": "single_feedback_transition_interventions",
        "provenance": evaluation_provenance(
            config_path=args.config,
            config=cfg,
            weight_identity=weights,
            device=device,
            seeds={"torch": args.seed},
            evaluation_data_dir=evaluation_data_dir,
        ),
        "result": result,
    }
    print(render_or_write_json(document, args.output))


if __name__ == "__main__":
    main()
