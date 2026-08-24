#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.manifest import file_sha256
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.evaluation.recurrent import evaluate_recurrent_continuation
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.training.checkpoint import load_checkpoint_for_evaluation
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant
from tiny_mistral_mptt.variants.bank import BankVariant
from tiny_mistral_mptt.variants.bank_add_hybrid import BankAddHybridVariant
from tiny_mistral_mptt.variants.bank_recirculation_hybrid import (
    BankRecirculationHybridVariant,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-force held-out continuations and compare exact cached "
            "K-pass inference against collapsed one-stream recurrence."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--prefill-passes",
        type=int,
        nargs="+",
        default=[2],
        help="one or more positive prompt-refinement depths K",
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
    args = parser.parse_args()

    if any(passes < 1 for passes in args.prefill_passes):
        raise SystemExit("--prefill-passes values must be positive")

    cfg = load_experiment_config(args.config)
    if cfg.variant not in {
        "memory_add",
        "recirculation",
        "bank",
        "bank_multiscale",
        "bank_add_hybrid",
        "bank_recirculation_hybrid",
    }:
        raise SystemExit("evaluate_recurrent_inference requires a cached recurrent variant")
    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    if not isinstance(
        model,
        (
            MemoryAddVariant,
            RecirculationVariant,
            BankVariant,
            BankAddHybridVariant,
            BankRecirculationHybridVariant,
        ),
    ):
        raise SystemExit("loaded variant does not implement recurrent memory inference")

    expected = file_sha256(f"{cfg.data_dir}/manifest.json")
    load_checkpoint_for_evaluation(
        args.checkpoint,
        model=model,
        expected_manifest_sha256=expected,
        expected_experiment_config=cfg.to_dict(),
    )
    model.eval()

    dataset = load_packed_dataset_for_experiment(cfg.data_dir, "validation", memory_write_mode=cfg.memory_write_mode, memory_write_stride=cfg.memory_write_stride)
    results = []
    for passes in args.prefill_passes:
        result = evaluate_recurrent_continuation(
            model,
            dataset,
            device=device,
            prefill_passes=passes,
            prompt_tokens=args.prompt_tokens,
            continuation_tokens=args.continuation_tokens,
            max_blocks=args.max_blocks,
            horizons=args.horizons,
        )
        results.append(asdict(result))

    document = {
        "variant": cfg.variant,
        "checkpoint": str(args.checkpoint),
        "prefill_passes": list(args.prefill_passes),
        "results": results,
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
