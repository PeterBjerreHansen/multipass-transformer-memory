#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.manifest import file_sha256
from tiny_mistral_mptt.evaluation.drift import parameter_drift_summary
from tiny_mistral_mptt.evaluation.provenance import (
    evaluation_provenance,
    load_evaluation_weights,
    render_or_write_json,
    seed_evaluation,
)
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.training.checkpoint import load_model_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure parameter drift from an architecture-compatible reference."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seed_evaluation(args.seed)
    cfg = load_experiment_config(args.config)
    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    added_ids = {id(parameter) for parameter in model.added_parameters()}
    added_names = {
        name for name, parameter in model.named_parameters() if id(parameter) in added_ids
    }

    reference_metadata = load_model_weights(
        args.reference,
        model=model,
        expected_experiment_config=cfg.to_dict(),
    )
    reference = {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
    }
    target_identity = load_evaluation_weights(
        model=model,
        config=cfg,
        checkpoint=args.checkpoint,
        initialized_baseline=False,
    )
    current = {
        name: parameter.detach().cpu().float()
        for name, parameter in model.named_parameters()
    }
    summary = parameter_drift_summary(
        reference,
        current,
        added_names=added_names,
    )
    document = {
        "evaluation_kind": "parameter_drift",
        "reference": {
            "path": str(Path(args.reference).resolve()),
            "sha256": file_sha256(args.reference),
            "metadata": reference_metadata,
        },
        "provenance": evaluation_provenance(
            config_path=args.config,
            config=cfg,
            weight_identity=target_identity,
            device=device,
            seeds={"torch": args.seed},
        ),
        "result": summary,
    }
    print(render_or_write_json(document, args.output))


if __name__ == "__main__":
    main()
