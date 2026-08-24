#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.manifest import file_sha256
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.evaluation.nll import evaluate_nll
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.training.checkpoint import load_checkpoint_for_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="experiment checkpoint generation; omit to evaluate initialized weights",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="pass depth for multipass NLL; defaults to the explicit pass-1 metric",
    )
    parser.add_argument("--max-blocks", type=int, default=None)
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    if args.checkpoint:
        expected = file_sha256(f"{cfg.data_dir}/manifest.json")
        load_checkpoint_for_evaluation(
            args.checkpoint,
            model=model,
            expected_manifest_sha256=expected,
            expected_experiment_config=cfg.to_dict(),
        )
    dataset = load_packed_dataset_for_experiment(cfg.data_dir, "validation", memory_write_mode=cfg.memory_write_mode, memory_write_stride=cfg.memory_write_stride)
    result = evaluate_nll(
        model,
        dataset,
        device=device,
        passes=args.passes,
        max_blocks=args.max_blocks,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
