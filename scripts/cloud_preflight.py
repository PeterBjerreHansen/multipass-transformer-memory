#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral.loading import verify_target_checkpoint
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.manifest import file_sha256, verify_artifact
from tiny_mistral_mptt.data.packed_dataset import memory_token_physical_length
from tiny_mistral_mptt.training.checkpoint import (
    candidate_checkpoint_paths,
    validate_checkpoint,
)
from tiny_mistral_mptt.training.provenance import hardware_provenance, source_provenance


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _occupied_run(output_dir: Path) -> bool:
    return any(
        path.exists()
        for path in (
            output_dir / "run.json",
            output_dir / "metrics.jsonl",
            output_dir / "segments.jsonl",
            output_dir / "checkpoints",
        )
    )


def _source_identity(source: dict | None) -> tuple[object, object]:
    source = source or {}
    return source.get("source_code_sha256"), source.get("uv_lock_sha256")


def _estimate_checkpoint_bytes(cfg, model_verification: dict | None) -> int | None:
    """Conservatively estimate one optimizer checkpoint generation."""
    if not model_verification or not model_verification.get("parameter_count"):
        return None
    parameter_bytes = {"float32": 4, "bfloat16": 2, "float16": 2}.get(cfg.dtype)
    if parameter_bytes is None:
        return None
    # Model weights plus two Adam moments, with head/metadata overhead. The
    # margin covers added Memory Attention parameters and non-tensor checkpoint state.
    tensor_bytes = int(model_verification["parameter_count"]) * parameter_bytes * 3
    return int(tensor_bytes * 1.25)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provider-agnostic preflight for a CUDA training run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--mode", choices=("new", "resume", "auto"), default="new")
    parser.add_argument(
        "--persistent-root",
        default=None,
        help="require output_dir to live underneath this persistent filesystem root",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = _repo_path(root, args.config)
    cfg = load_experiment_config(config_path)
    failures: list[str] = []
    warnings: list[str] = []

    try:
        device = resolve_device(cfg.device)
    except RuntimeError as exc:
        device = None
        failures.append(str(exc))

    source = source_provenance(root)
    if source["git_commit"] is None:
        warnings.append("git commit unavailable; source_code_sha256 identifies execution code")
    if source["git_dirty"] and not args.allow_dirty:
        failures.append("git worktree is dirty")
    if device is None or device.type != "cuda":
        failures.append(f"cloud preflight requires CUDA; resolved {device}")
    if (
        device is not None
        and device.type == "cuda"
        and cfg.autocast_dtype == "bfloat16"
        and not torch.cuda.is_bf16_supported()
    ):
        failures.append("config requests BF16 autocast but GPU reports no BF16 support")

    model_dir = _repo_path(root, cfg.model_dir)
    model_verification = None
    if not model_dir.exists():
        failures.append(f"model_dir does not exist: {model_dir}")
    else:
        try:
            model_verification = verify_target_checkpoint(model_dir)
            if not model_verification["ok"]:
                failures.append("model checkpoint does not match the pinned TinyMistral target")
        except Exception as exc:
            failures.append(f"model checkpoint verification failed: {exc}")

    data_dir = _repo_path(root, cfg.data_dir)
    manifest_sha256 = None
    data_manifest = None
    if not data_dir.exists():
        failures.append(f"data_dir does not exist: {data_dir}")
    else:
        try:
            data_manifest = verify_artifact(data_dir)
            manifest_sha256 = file_sha256(data_dir / "manifest.json")
        except Exception as exc:
            failures.append(f"data artifact verification failed: {exc}")

    checkpoint_hashes: dict[str, str | None] = {
        "init_from_sha256": None,
        "resume_from_sha256": None,
    }
    if cfg.init_from:
        init_path = _repo_path(root, cfg.init_from)
        if not init_path.exists():
            failures.append(f"init_from checkpoint does not exist: {cfg.init_from}")
        else:
            checkpoint_hashes["init_from_sha256"] = file_sha256(init_path)
    if cfg.resume_from:
        resume_path = _repo_path(root, cfg.resume_from)
        if not resume_path.exists():
            failures.append(f"resume_from checkpoint does not exist: {cfg.resume_from}")
        else:
            checkpoint_hashes["resume_from_sha256"] = file_sha256(resume_path)

    output_dir = _repo_path(root, cfg.output_dir)
    occupied = _occupied_run(output_dir)
    run_json_exists = (output_dir / "run.json").exists()
    effective_mode = args.mode
    if args.mode == "auto":
        if not occupied:
            effective_mode = "new"
        elif run_json_exists:
            effective_mode = "resume"
        else:
            failures.append("auto mode found run artifacts without run.json; refusing ambiguous recovery")
            effective_mode = "invalid"

    selected_checkpoint = None
    selected_checkpoint_metadata = None
    checkpoint_errors: list[str] = []
    if effective_mode == "new":
        if occupied:
            failures.append(f"new mode refuses existing run artifacts: {output_dir}")
    elif effective_mode == "resume":
        if not run_json_exists:
            failures.append(f"resume mode requires run.json: {output_dir}")
        for path in candidate_checkpoint_paths(output_dir):
            try:
                metadata = validate_checkpoint(
                    path,
                    expected_manifest_sha256=manifest_sha256,
                    expected_experiment_config=cfg.to_dict(),
                    expected_source_provenance=source,
                    allow_source_mismatch=args.allow_source_mismatch,
                )
            except Exception as exc:
                checkpoint_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            selected_checkpoint = path
            selected_checkpoint_metadata = metadata
            break
        if selected_checkpoint is None:
            failures.append(
                "resume mode found no readable checkpoint generation"
                + (f" ({'; '.join(checkpoint_errors)})" if checkpoint_errors else "")
            )
        else:
            recorded_source = selected_checkpoint_metadata.get("source_provenance")
            if (
                recorded_source
                and recorded_source.get("git_commit")
                and source.get("git_commit")
                and recorded_source.get("git_commit") != source.get("git_commit")
                and _source_identity(recorded_source) == _source_identity(source)
            ):
                warnings.append("Git commit differs but execution-code and uv.lock hashes are identical")

    persistent_report = None
    if args.persistent_root:
        persistent_root = _repo_path(root, args.persistent_root)
        if not persistent_root.exists():
            failures.append(f"persistent root does not exist: {persistent_root}")
        elif not _inside(output_dir, persistent_root):
            failures.append(
                f"output_dir is not underneath persistent root: output={output_dir} root={persistent_root}"
            )
        else:
            probe = output_dir if output_dir.exists() else output_dir.parent
            usage = shutil.disk_usage(probe)
            generation_sizes = [
                path.stat().st_size
                for path in candidate_checkpoint_paths(output_dir)
                if path.exists()
            ]
            estimated = _estimate_checkpoint_bytes(cfg, model_verification)
            checkpoint_bytes = max(
                [*generation_sizes, *([] if estimated is None else [estimated])],
                default=None,
            )
            required_free = 3 * checkpoint_bytes if checkpoint_bytes else None
            if required_free is not None and usage.free < required_free:
                failures.append(
                    "insufficient free space for previous/current/new checkpoint rotation"
                )
            if required_free is None:
                warnings.append(
                    "checkpoint size could not be estimated; free-space check is unavailable"
                )
            persistent_report = {
                "root": str(persistent_root),
                "free_bytes": int(usage.free),
                "estimated_checkpoint_bytes": checkpoint_bytes,
                "recommended_free_bytes": required_free,
            }

    batching = None
    if data_manifest is not None:
        linguistic_length = int(data_manifest.sequence_length)
        if cfg.memory_write_mode == "memory_token":
            assert cfg.memory_write_stride is not None
            physical_length = memory_token_physical_length(
                linguistic_length, int(cfg.memory_write_stride)
            )
        else:
            physical_length = linguistic_length
        micro_tokens = cfg.batch_size * linguistic_length
        micro_positions = cfg.batch_size * physical_length
        batching = {
            "linguistic_sequence_length": linguistic_length,
            "physical_sequence_length": physical_length,
            "control_positions_per_sequence": physical_length - linguistic_length,
            "microbatch_size": cfg.batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "microbatch_tokens": micro_tokens,
            "microbatch_model_positions": micro_positions,
            "nominal_optimizer_batch_tokens": micro_tokens * cfg.grad_accum_steps,
            "nominal_optimizer_batch_model_positions": micro_positions * cfg.grad_accum_steps,
        }

    report = {
        "status": "pass" if not failures else "fail",
        "config": str(config_path),
        "requested_mode": args.mode,
        "effective_mode": effective_mode,
        "source": source,
        "precision": {"parameter_dtype": cfg.dtype, "autocast_dtype": cfg.autocast_dtype},
        "memory_attention": {
            "variant": cfg.variant,
            "memory_window": cfg.memory_window,
            "memory_write_mode": cfg.memory_write_mode,
            "memory_write_stride": cfg.memory_write_stride,
            "memory_token_visibility": cfg.memory_token_visibility,
            "memory_layers": cfg.memory_layers,
            "memory_position_encoding": cfg.memory_position_encoding,
            "memory_dense_window": cfg.memory_dense_window,
            "memory_sparse_window": cfg.memory_sparse_window,
            "memory_sparse_stride": cfg.memory_sparse_stride,
        },
        "strided_attention": {
            "sparse_attention_stride": cfg.sparse_attention_stride,
            "sparse_attention_window": cfg.sparse_attention_window,
            "sparse_attention_layers": cfg.sparse_attention_layers,
        },
        "batching": batching,
        "model_verification": model_verification,
        "hardware": hardware_provenance(device),
        "data_manifest_sha256": manifest_sha256,
        "checkpoint_hashes": checkpoint_hashes,
        "selected_checkpoint": str(selected_checkpoint) if selected_checkpoint else None,
        "persistent_storage": persistent_report,
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
