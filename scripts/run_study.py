#!/usr/bin/env python
"""Validate, wire, and sequentially execute a colocated benchmark study."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

from tiny_mistral.device import resolve_device, synchronize
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.precision import autocast_context
from tiny_mistral_mptt.studies import StudyValidationError, verify_study
from tiny_mistral_mptt.training.checkpoint import load_model_weights
from tiny_mistral_mptt.training.phases import configure_phase


def _load_arm_configs(study_dir: Path) -> dict[str, Path]:
    raw = yaml.safe_load((study_dir / "STUDY.yaml").read_text(encoding="utf-8")) or {}
    return {
        str(arm["id"]): (study_dir / str(arm["config"])).resolve()
        for arm in raw.get("arms", [])
    }


def _output_dir_for_config(config_path: Path, *, root: Path) -> Path:
    """Resolve a config's output directory in the same way training does."""
    output_dir = Path(load_experiment_config(config_path).output_dir)
    return output_dir if output_dir.is_absolute() else root / output_dir


def _has_training_trajectory(config_path: Path, *, root: Path) -> bool:
    """Return whether an output directory contains a trajectory to recover.

    ``Trainer`` treats a run journal as authoritative and refuses ambiguous
    partial artifacts.  Looking for either the journal or the metrics/segment
    logs here lets the launcher select ``--resume-auto`` only for an existing
    trajectory, while preserving ``init_from`` for a fresh Phase-B run.
    """
    output_dir = _output_dir_for_config(config_path, root=root)
    return any(
        (output_dir / filename).is_file()
        for filename in ("run.json", "metrics.jsonl", "segments.jsonl")
    ) or (output_dir / "checkpoints").is_dir()


def _should_resume_auto(config_path: Path, *, root: Path) -> bool:
    """Choose automatic recovery for this config without overriding init-from."""
    cfg = load_experiment_config(config_path)
    if cfg.resume_from is not None:
        return False
    if cfg.init_from is not None and not _has_training_trajectory(config_path, root=root):
        return False
    return True


def _wire_arm(config_path: Path, *, wire_device: str | None) -> None:
    cfg = load_experiment_config(config_path)
    if wire_device is not None:
        cfg = replace(cfg, device=wire_device)
        cfg.validate()

    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    if cfg.init_from is not None:
        init_path = Path(cfg.init_from)
        if not init_path.is_absolute():
            init_path = Path(__file__).resolve().parents[1] / init_path
        if not init_path.is_file():
            raise FileNotFoundError(f"wiring init_from does not exist: {init_path}")
        load_model_weights(
            init_path,
            model=model,
            expected_experiment_config=cfg.to_dict(),
        )
    train_data = load_packed_dataset_for_experiment(
        cfg.data_dir,
        "train",
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
    )
    configure_phase(model, cfg.phase)
    model.train()
    pass_counts = sorted(
        {
            passes
            for stage in cfg.normalized_pass_schedule()
            for passes in stage["probabilities"]
        }
    )
    input_ids = train_data.batch([0], device=device)
    for passes in pass_counts:
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        with autocast_context(device, cfg.autocast_dtype):
            output = model.compute_training_loss(
                input_ids,
                training_forward=cfg.training_forward,
                phase=cfg.phase,
                passes=passes,
                loss_weights=cfg.ntp_loss_weights_for_passes(passes),
                activation_checkpointing=cfg.recirculation_activation_checkpointing,
            )
        if not bool(torch.isfinite(output.loss.detach()).item()):
            raise RuntimeError(
                f"non-finite wiring loss for {config_path} "
                f"forward={cfg.training_forward} K={passes}"
            )
        output.loss.backward()
        synchronize(device)
        elapsed_seconds = time.perf_counter() - started
        active_gradient_parameter_elements = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and bool(parameter.grad.detach().ne(0).any().item())
        )
        if active_gradient_parameter_elements == 0:
            raise RuntimeError(
                f"wiring produced no nonzero gradients for {config_path} "
                f"forward={cfg.training_forward} K={passes}"
            )
        memory = ""
        if device.type == "cuda":
            memory = (
                f" peak_allocated_gib={torch.cuda.max_memory_allocated(device) / 2**30:.3f}"
                f" peak_reserved_gib={torch.cuda.max_memory_reserved(device) / 2**30:.3f}"
            )
        print(
            f"PASS: wired {config_path.stem} device={device} "
            f"forward={cfg.training_forward} K={passes} "
            f"loss={float(output.loss.detach().cpu()):.6f} "
            f"active_gradient_parameter_elements={active_gradient_parameter_elements} "
            f"elapsed_seconds={elapsed_seconds:.3f}{memory}"
        )

    del output, input_ids, train_data, model
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, wire, and sequentially run a benchmark study."
    )
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        help="run only this arm; repeat for multiple arms (default: all arms)",
    )
    parser.add_argument("--skip-wire", action="store_true")
    parser.add_argument("--wire-only", action="store_true")
    parser.add_argument(
        "--wire-device",
        choices=("cpu", "mps", "cuda"),
        default=None,
    )
    parser.add_argument("--no-resume-auto", action="store_true")
    parser.add_argument("--until-unique-tokens", type=int, default=None)
    parser.add_argument("--allow-source-mismatch", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    study_dir = args.study_dir if args.study_dir.is_absolute() else root / args.study_dir
    try:
        verification = verify_study(study_dir)
    except StudyValidationError as exc:
        raise SystemExit(f"study validation failed: {exc}") from exc

    configs = _load_arm_configs(study_dir)
    selected = list(verification.arm_ids if args.arms is None else args.arms)
    unknown = [arm_id for arm_id in selected if arm_id not in configs]
    if unknown:
        raise SystemExit(
            f"unknown arm(s): {unknown}; available arms: {list(verification.arm_ids)}"
        )

    print(f"PASS: verified {verification.name} arms={','.join(selected)}")
    if not args.skip_wire:
        for arm_id in selected:
            _wire_arm(configs[arm_id], wire_device=args.wire_device)
    if args.wire_only:
        return

    train_script = root / "scripts" / "train.py"
    for arm_id in selected:
        command = [
            sys.executable,
            str(train_script),
            "--config",
            str(configs[arm_id]),
        ]
        if not args.no_resume_auto and _should_resume_auto(configs[arm_id], root=root):
            command.append("--resume-auto")
        if args.until_unique_tokens is not None:
            command.extend(["--until-unique-tokens", str(args.until_unique_tokens)])
        if args.allow_source_mismatch:
            command.append("--allow-source-mismatch")
        print(f"START: {arm_id} {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=root, check=True)
        print(f"PASS: completed {arm_id}", flush=True)


if __name__ == "__main__":
    main()
