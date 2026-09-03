#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.evaluation.pass_depth import evaluate_pass_depth
from tiny_mistral_mptt.evaluation.provenance import (
    add_checkpoint_arguments,
    evaluation_provenance,
    load_evaluation_weights,
    render_or_write_json,
    seed_evaluation,
)
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.evaluation.settings import add_execution_arguments, resolve_evaluation_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exact full-sequence validation NLL and state stability "
            "for every prompt-independent pass from K=1 through K=max."
        )
    )
    parser.add_argument("--config", required=True)
    add_checkpoint_arguments(parser)
    add_execution_arguments(parser)
    parser.add_argument("--evaluation-data-dir", default=None)
    parser.add_argument("--passes", type=int, default=None, help="default: experiment eval_passes")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seed_evaluation(args.seed)
    cfg = load_experiment_config(args.config)
    device = resolve_device(cfg.device if args.device is None else args.device)
    model = load_variant_from_config(cfg, device=device)
    settings = resolve_evaluation_settings(
        cfg, model, passes=args.passes, autocast_dtype=args.autocast_dtype,
    )
    weights = load_evaluation_weights(
        model=model,
        config=cfg,
        checkpoint=args.checkpoint,
        initialized_baseline=args.initialized_baseline,
    )
    evaluation_data_dir = args.evaluation_data_dir or cfg.data_dir
    dataset = load_packed_dataset_for_experiment(
        evaluation_data_dir,
        "validation",
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
    )
    result = evaluate_pass_depth(
        model,
        dataset,
        device=device,
        passes=settings.passes,
        max_blocks=args.max_blocks,
        autocast_dtype=settings.autocast_dtype,
    )
    document = {
        "schema_version": 2,
        "evaluation_kind": "full_sequence_pass_depth_convergence",
        "semantics": {
            "passes": list(range(1, settings.passes + 1)),
            "teacher_forced": True,
            "generation": False,
        },
        "resolved_settings": asdict(settings),
        "provenance": evaluation_provenance(
            config_path=args.config,
            config=cfg,
            weight_identity=weights,
            device=device,
            seeds={"torch": args.seed},
            evaluation_data_dir=evaluation_data_dir,
        ),
        "result": asdict(result),
    }
    print(render_or_write_json(document, args.output))


if __name__ == "__main__":
    main()
