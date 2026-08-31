#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.evaluation.pass_depth import evaluate_pass_depth
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.variants.fbt import FBTVariant
from tiny_mistral_mptt.variants.multipass import shift_previous_hidden


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.float().pow(2).mean().sqrt().cpu())


@torch.no_grad()
def _fusion_stats(
    model: FBTVariant,
    token_embeddings: torch.Tensor,
    previous_hidden: torch.Tensor,
    *,
    normalize_gate_input: bool,
) -> dict[str, float]:
    model.normalize_gate_input = normalize_gate_input
    gate_input = (
        model.feedback_input_norm(token_embeddings)
        if normalize_gate_input
        else token_embeddings
    )
    gate_logits = model.feedback_gate(gate_input)
    shifted = shift_previous_hidden(previous_hidden)
    fused_pre_norm = model.feedback_value(shifted) * torch.sigmoid(gate_logits)
    fused_input = model.feedback_inputs(token_embeddings, previous_hidden)
    return {
        "gate_input_rms": _rms(gate_input),
        "gate_logit_std": float(gate_logits.float().std().cpu()),
        "fused_pre_norm_rms": _rms(fused_pre_norm[:, 1:, :]),
        "fused_input_rms": _rms(fused_input[:, 1:, :]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare untouched current and paper-style FBT fusion on fixed validation blocks."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--max-blocks", type=int, default=32)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    if cfg.variant != "fbt":
        raise SystemExit("diagnose_fusion requires variant=fbt")
    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    if not isinstance(model, FBTVariant):
        raise RuntimeError("model factory did not construct FBTVariant")
    dataset = load_packed_dataset_for_experiment(cfg.data_dir, "validation")

    model.eval()
    ids = dataset.batch([0], device=device)
    with torch.no_grad():
        token_embeddings = model.input_embeddings(ids)
        previous_hidden = model.compute_passes(ids, passes=1).passes[0].hidden_states

    report: dict[str, object] = {
        "config": str(args.config),
        "validation_blocks": min(len(dataset), args.max_blocks),
        "passes": args.passes,
        "embedding_rms": _rms(token_embeddings),
        "first_pass_hidden_rms": _rms(previous_hidden),
        "training_jitter_std": {"current": 0.0, "paper": 0.02},
        "modes": {},
    }
    modes = report["modes"]
    assert isinstance(modes, dict)
    for name, normalize_gate_input in (("current", False), ("paper", True)):
        stats = _fusion_stats(
            model,
            token_embeddings,
            previous_hidden,
            normalize_gate_input=normalize_gate_input,
        )
        result = evaluate_pass_depth(
            model,
            dataset,
            device=device,
            passes=args.passes,
            max_blocks=args.max_blocks,
        )
        modes[name] = {
            **stats,
            "nll_by_pass": list(result.nll_by_pass),
            "hidden_delta_rms": list(result.hidden_delta_rms),
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
