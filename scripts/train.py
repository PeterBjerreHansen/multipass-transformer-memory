#!/usr/bin/env python
from __future__ import annotations

import argparse
import signal
from pathlib import Path

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.training.trainer import Trainer


_STOP_REQUESTED = False


def _request_stop(signum, frame) -> None:  # pragma: no cover - OS integration
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _has_training_trajectory(output_dir: str) -> bool:
    path = Path(output_dir)
    return any(
        (path / filename).is_file()
        for filename in ("run.json", "metrics.jsonl", "segments.jsonl")
    ) or (path / "checkpoints").is_dir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a TinyMistral continued-pretraining experiment stage."
    )
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume-from",
        default=None,
        help="exactly resume optimizer/RNG/data/pass-scheduler state from one checkpoint",
    )
    group.add_argument(
        "--resume-auto",
        action="store_true",
        help="start if output_dir is empty; otherwise resume the newest valid generation",
    )
    group.add_argument(
        "--init-from",
        default=None,
        help="load model weights only and begin a fresh run",
    )
    parser.add_argument(
        "--allow-source-mismatch",
        action="store_true",
        help="development-only escape hatch for resuming with different execution code/uv.lock",
    )
    parser.add_argument("--until-unique-tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    resume_auto = bool(args.resume_auto)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
        cfg.init_from = None
    elif resume_auto:
        if cfg.init_from is not None and not _has_training_trajectory(cfg.output_dir):
            # ``init_from`` is the correct first-run behavior for a config
            # whose output directory is empty, even when a caller requested
            # the convenient auto mode.
            resume_auto = False
        else:
            # Once a trajectory exists, automatic recovery must take
            # precedence over one-time model initialisation.
            cfg.resume_from = None
            # Keep the original initialization parent in the config. Trainer
            # loads the newest run checkpoint after constructing the model.
    elif args.init_from is not None:
        cfg.init_from = args.init_from
        cfg.resume_from = None
    cfg.validate()

    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    train_data = load_packed_dataset_for_experiment(
        cfg.data_dir,
        "train",
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
    )
    validation_data = load_packed_dataset_for_experiment(
        cfg.data_dir,
        "validation",
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
    )

    signal.signal(signal.SIGINT, _request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop)

    trainer = Trainer(
        model=model,
        config=cfg,
        train_data=train_data,
        validation_data=validation_data,
        device=device,
        resume_auto=resume_auto,
        allow_source_mismatch=args.allow_source_mismatch,
        stop_requested=lambda: _STOP_REQUESTED,
    )
    state = trainer.train(until_unique_tokens=args.until_unique_tokens)
    print(
        "PASS: training stopped " if _STOP_REQUESTED else "PASS: training completed ",
        f"phase={state.phase} steps={state.optimizer_steps} ",
        f"unique_tokens={state.unique_tokens_seen} ",
        f"model_positions={state.model_positions_seen} ",
        f"token_equivalent={state.token_equivalent_compute}",
        sep="",
    )


if __name__ == "__main__":
    main()
