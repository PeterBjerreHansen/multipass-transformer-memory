#!/usr/bin/env python
from __future__ import annotations

import argparse


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
from tiny_mistral_mptt.evaluation.interventions import evaluate_memory_interventions
from tiny_mistral_mptt.evaluation.settings import add_execution_arguments, resolve_evaluation_settings


SUPPORTED = {
    "memory_add",
    "recirculation",
    "recurrent_memory",
    "memory_attention",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run real/zero/mismatched recurrence interventions. The hybrid also "
            "reports recurrent and Memory Attention channel interventions independently."
        )
    )
    parser.add_argument("--config", required=True)
    add_checkpoint_arguments(parser)
    add_execution_arguments(parser)
    parser.add_argument("--evaluation-data-dir", default=None)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seed_evaluation(args.seed)
    cfg = load_experiment_config(args.config)
    if canonical_variant_name(cfg.variant) not in SUPPORTED:
        raise SystemExit(
            "evaluate_memory_interventions requires a recurrent-memory or Memory Attention variant"
        )
    device = resolve_device(cfg.device if args.device is None else args.device)
    model = load_variant_from_config(cfg, device=device)
    settings = resolve_evaluation_settings(cfg, model, autocast_dtype=args.autocast_dtype)
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
    result = evaluate_memory_interventions(
        model, dataset, device=device, max_blocks=args.max_blocks,
        autocast_dtype=settings.autocast_dtype,
    )
    result["variant"] = cfg.variant

    document = {
        "schema_version": 3,
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
