#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from tiny_mistral_mptt.data.config import load_data_config
from tiny_mistral_mptt.data.dolmino import prepare_dolmino


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a deterministic local Dolmino token artifact.")
    parser.add_argument(
        "--config",
        default="data/dolmino/wiring_2048/config.yaml",
        help="data recipe beside the artifact; defaults to the 5M wiring recipe",
    )
    args = parser.parse_args()
    cfg = load_data_config(args.config)
    manifest = prepare_dolmino(
        output_dir=cfg.output_dir,
        model_dir=cfg.model_dir,
        sequence_length=cfg.sequence_length,
        train_tokens=cfg.train_tokens,
        validation_tokens=cfg.validation_tokens,
        validation_skip_tokens=cfg.validation_skip_tokens,
        train_skip_tokens=cfg.train_skip_tokens,
        seed=cfg.seed,
        dataset_repo=cfg.dataset_repo,
        revision=cfg.revision,
        shuffle_buffer=cfg.shuffle_buffer,
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
