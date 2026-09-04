#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.evaluation.provenance import (
    add_checkpoint_arguments,
    evaluation_provenance,
    load_evaluation_weights,
    render_or_write_json,
    seed_evaluation,
)
from tiny_mistral_mptt.evaluation.recurrent import evaluate_recurrent_continuation
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.evaluation.settings import add_execution_arguments, resolve_evaluation_settings
from tiny_mistral_mptt.variants.multipass import MultiPassVariant


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-force evaluation continuations and compare exact cached "
            "K-pass inference against collapsed one-stream recurrence."
        )
    )
    parser.add_argument("--config", required=True)
    add_checkpoint_arguments(parser)
    add_execution_arguments(parser)
    parser.add_argument("--evaluation-data-dir", default=None)
    parser.add_argument(
        "--prefill-passes",
        type=int,
        nargs="+",
        default=None,
        help="prompt-refinement depths; default: experiment eval_prefill_passes or eval_passes",
    )
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--continuation-tokens", type=int, default=256)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        default=None,
        help="optional cumulative continuation horizons; default powers of two",
    )
    parser.add_argument("--output", default=None, help="optional JSON output path")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.prefill_passes is not None and any(passes < 1 for passes in args.prefill_passes):
        raise SystemExit("--prefill-passes values must be positive")

    seed_evaluation(args.seed)
    cfg = load_experiment_config(args.config)
    device = resolve_device(cfg.device if args.device is None else args.device)
    model = load_variant_from_config(cfg, device=device)
    if not isinstance(model, MultiPassVariant) or not model.supports_cached_feedback:
        raise SystemExit("loaded variant does not implement recurrent memory inference")

    settings = resolve_evaluation_settings(cfg, model, autocast_dtype=args.autocast_dtype)
    prefill_depths = args.prefill_passes or [settings.prefill_passes]
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
        verify_integrity=True,
    )
    results = []
    for passes in prefill_depths:
        result = evaluate_recurrent_continuation(
            model,
            dataset,
            device=device,
            prefill_passes=passes,
            prompt_tokens=args.prompt_tokens,
            continuation_tokens=args.continuation_tokens,
            max_blocks=args.max_blocks,
            horizons=args.horizons,
            autocast_dtype=settings.autocast_dtype,
        )
        results.append(asdict(result))

    document = {
        "schema_version": 2,
        "evaluation_kind": "exact_vs_feedback_continuation_diagnostic",
        "variant": cfg.variant,
        "prefill_passes": list(prefill_depths),
        "provenance": evaluation_provenance(
            config_path=args.config,
            config=cfg,
            weight_identity=weights,
            device=device,
            seeds={"torch": args.seed},
            evaluation_data_dir=evaluation_data_dir,
        ),
        "results": results,
    }
    rendered = render_or_write_json(document, args.output)
    print(rendered)


if __name__ == "__main__":
    main()
