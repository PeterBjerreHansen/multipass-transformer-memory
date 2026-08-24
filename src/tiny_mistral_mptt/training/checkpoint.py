from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import re
from typing import Any

import torch


FORMAT_VERSION = 3
_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d{12})\.pt$")


@dataclass
class TrainState:
    optimizer_steps: int = 0
    micro_steps: int = 0
    unique_tokens_seen: int = 0  # linguistic/data tokens only
    model_positions_seen: int = 0  # includes input-only control positions
    token_equivalent_compute: int = 0  # physical positions * effective passes
    phase: str = "B"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_state: dict,
    train_state: TrainState,
    experiment_config: dict,
    data_manifest_sha256: str,
    pass_scheduler_state: dict[str, Any] | None,
    source_provenance: dict[str, Any] | None,
    checkpoint_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler_state,
        "pass_scheduler": pass_scheduler_state,
        "train_state": asdict(train_state),
        "rng": capture_rng_state(),
        "experiment_config": experiment_config,
        "data_manifest_sha256": data_manifest_sha256,
        "source_provenance": source_provenance,
        "checkpoint_metadata": checkpoint_metadata or {},
    }


def _save_payload_durable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_state: dict,
    train_state: TrainState,
    experiment_config: dict,
    data_manifest_sha256: str,
    pass_scheduler_state: dict[str, Any] | None = None,
    source_provenance: dict[str, Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> Path:
    """Durably write one explicit checkpoint path.

    Training runs use generation checkpoints below; this small entry point is
    useful for tests and explicit initialization artifacts.
    """
    path = Path(path)
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        sampler_state=sampler_state,
        train_state=train_state,
        experiment_config=experiment_config,
        data_manifest_sha256=data_manifest_sha256,
        pass_scheduler_state=pass_scheduler_state,
        source_provenance=source_provenance,
        checkpoint_metadata=checkpoint_metadata,
    )
    _save_payload_durable(path, payload)
    return path


def checkpoint_filename(unique_tokens_seen: int) -> str:
    tokens = int(unique_tokens_seen)
    if tokens < 0:
        raise ValueError("unique_tokens_seen must be non-negative")
    return f"checkpoint_{tokens:012d}.pt"


def checkpoint_directory(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoints"


def discover_checkpoint_generations(run_dir: str | Path) -> list[Path]:
    directory = checkpoint_directory(run_dir)
    if not directory.exists():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = _CHECKPOINT_RE.match(path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in candidates]


def _read_latest_pointer(run_dir: str | Path) -> dict[str, Any] | None:
    path = checkpoint_directory(run_dir) / "latest.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def candidate_checkpoint_paths(run_dir: str | Path) -> list[Path]:
    directory = checkpoint_directory(run_dir)
    pointer = _read_latest_pointer(run_dir)
    ordered: list[Path] = []
    if pointer:
        for key in ("current", "previous"):
            name = pointer.get(key)
            if isinstance(name, str):
                path = directory / name
                if path not in ordered:
                    ordered.append(path)
    for path in discover_checkpoint_generations(run_dir):
        if path not in ordered:
            ordered.append(path)
    return ordered


def _require_payload(payload: dict[str, Any]) -> None:
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported experiment checkpoint format; expected v{FORMAT_VERSION}"
        )
    required = {
        "model",
        "optimizer",
        "sampler",
        "train_state",
        "rng",
        "experiment_config",
        "data_manifest_sha256",
        "source_provenance",
        "checkpoint_metadata",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint missing required fields: {missing}")
    # Construction validates the complete clean-break counter schema.
    TrainState(**payload["train_state"])


def _checkpoint_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    state = TrainState(**payload["train_state"])
    return {
        "format_version": FORMAT_VERSION,
        "train_state": asdict(state),
        "experiment_config": payload["experiment_config"],
        "data_manifest_sha256": payload["data_manifest_sha256"],
        "source_provenance": payload["source_provenance"],
        "checkpoint_metadata": payload["checkpoint_metadata"],
    }


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    _require_payload(payload)
    return _checkpoint_metadata(payload)


def save_checkpoint_generation(
    run_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler_state: dict,
    train_state: TrainState,
    experiment_config: dict,
    data_manifest_sha256: str,
    pass_scheduler_state: dict[str, Any] | None = None,
    source_provenance: dict[str, Any] | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    keep_last: int = 2,
) -> Path:
    """Commit one resumable generation, optionally retaining a fallback."""
    keep_last = int(keep_last)
    if keep_last < 1:
        raise ValueError("generation checkpointing requires keep_last>=1")
    directory = checkpoint_directory(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / checkpoint_filename(train_state.unique_tokens_seen)
    payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        sampler_state=sampler_state,
        train_state=train_state,
        experiment_config=experiment_config,
        data_manifest_sha256=data_manifest_sha256,
        pass_scheduler_state=pass_scheduler_state,
        source_provenance=source_provenance,
        checkpoint_metadata=checkpoint_metadata,
    )
    _save_payload_durable(path, payload)

    # Verify the new generation before advertising it as current.
    metadata = inspect_checkpoint(path)
    if int(metadata["train_state"]["unique_tokens_seen"]) != train_state.unique_tokens_seen:
        raise RuntimeError("checkpoint verification returned the wrong token count")

    generations = discover_checkpoint_generations(run_dir)
    previous = None
    if keep_last >= 2:
        previous = next(
            (candidate.name for candidate in generations if candidate != path), None
        )
    pointer = {
        "format_version": 1,
        "current": path.name,
        "previous": previous,
        "unique_tokens_seen": int(train_state.unique_tokens_seen),
        "model_positions_seen": int(train_state.model_positions_seen),
        "optimizer_steps": int(train_state.optimizer_steps),
    }
    _durable_replace_bytes(
        directory / "latest.json",
        (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )

    # Prune only after the new payload and pointer are durable.
    for candidate in generations[keep_last:]:
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(directory)
    return path


def _resume_config_view(config: dict[str, Any]) -> dict[str, Any]:
    # Canonicalize the pre-NMP generic NTP pass-weight names and fill defaults
    # for fields introduced after older checkpoints were written. This keeps
    # exact resume viable while making new configs self-describing.
    canonical = dict(config)
    if "ntp_pass_loss_weights" not in canonical:
        canonical["ntp_pass_loss_weights"] = canonical.get("pass_loss_weights")
    if "ntp_pass_loss_weights_by_k" not in canonical:
        canonical["ntp_pass_loss_weights_by_k"] = canonical.get("pass_loss_weights_by_k")
    canonical.setdefault("recurrent_nmp_target_normalization", "rms")
    canonical.setdefault("recurrent_nmp_pass_loss_weights_by_k", None)
    canonical.setdefault("bank_nmp_pass_loss_weights_by_k", None)
    canonical.pop("pass_loss_weights", None)
    canonical.pop("pass_loss_weights_by_k", None)
    ignored = {
        "model_dir",
        "data_dir",
        "output_dir",
        "resume_from",
        "init_from",
        "eval_every_tokens",
        "eval_batches",
        "eval_passes",
        "checkpoint_every_tokens",
        "checkpoint_every_seconds",
        "checkpoint_keep_last",
        "snapshot_at_tokens",
    }
    schedule = canonical.get("lr_schedule")
    schedule_type = "cosine" if schedule is None else str(schedule.get("type", "cosine"))
    if schedule_type == "constant":
        ignored.add("max_unique_tokens")
    return {key: value for key, value in canonical.items() if key not in ignored}


def _source_identity(source: dict[str, Any] | None) -> tuple[Any, Any]:
    if not source:
        return (None, None)
    return (source.get("source_code_sha256"), source.get("uv_lock_sha256"))


def _changed_experiment_config_fields(
    recorded_config: dict[str, Any], requested_config: dict[str, Any]
) -> list[str]:
    recorded = _resume_config_view(recorded_config)
    requested = _resume_config_view(requested_config)
    return sorted(
        key
        for key in set(recorded) | set(requested)
        if recorded.get(key) != requested.get(key)
    )


def _validate_payload(
    payload: dict[str, Any],
    *,
    expected_manifest_sha256: str | None,
    expected_experiment_config: dict[str, Any] | None,
    expected_source_provenance: dict[str, Any] | None,
    allow_source_mismatch: bool,
    pass_scheduler=None,
) -> None:
    _require_payload(payload)
    if (
        expected_manifest_sha256 is not None
        and payload["data_manifest_sha256"] != expected_manifest_sha256
    ):
        raise ValueError("data manifest changed across resume")
    if expected_experiment_config is not None:
        changed = _changed_experiment_config_fields(
            payload["experiment_config"], expected_experiment_config
        )
        if changed:
            raise ValueError(f"experiment config changed across resume: {changed}")
    if expected_source_provenance is not None and not allow_source_mismatch:
        if _source_identity(payload["source_provenance"]) != _source_identity(expected_source_provenance):
            raise ValueError("execution-code or uv.lock hash changed across resume")
    if pass_scheduler is not None:
        scheduler_state = payload.get("pass_scheduler")
        if scheduler_state is None:
            raise ValueError("checkpoint is missing pass scheduler state")
        recorded_stages = scheduler_state.get("stages")
        if recorded_stages is not None and recorded_stages != pass_scheduler.stages:
            raise ValueError("pass schedule changed across resume")


def load_model_weights(path: str | Path, *, model: torch.nn.Module) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    _require_payload(payload)
    checkpoint_state = payload["model"]
    expected_keys = set(model.state_dict())
    checkpoint_keys = set(checkpoint_state)
    missing = expected_keys - checkpoint_keys
    unexpected = checkpoint_keys - expected_keys

    allowed_missing: set[str] = set()
    prefixes = tuple(
        getattr(model, "initialization_only_state_prefixes", lambda: ())()
    )
    for prefix in prefixes:
        expected_for_head = {key for key in expected_keys if key.startswith(prefix)}
        present_for_head = {key for key in checkpoint_keys if key.startswith(prefix)}
        if present_for_head and present_for_head != expected_for_head:
            absent = sorted(expected_for_head - present_for_head)
            extra = sorted(present_for_head - expected_for_head)
            raise RuntimeError(
                f"init_from checkpoint contains a partial {prefix!r} module; "
                f"missing={absent}, unexpected={extra}"
            )
        if not present_for_head:
            allowed_missing.update(expected_for_head)

    disallowed_missing = sorted(missing - allowed_missing)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "init_from model state is incompatible; "
            f"missing={disallowed_missing}, unexpected={sorted(unexpected)}"
        )
    result = model.load_state_dict(checkpoint_state, strict=False)
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            "init_from compatibility check disagreed with PyTorch state loading"
        )
    return {
        "source_path": str(path),
        "source_train_state": payload["train_state"],
        "source_experiment_config": payload["experiment_config"],
        "freshly_initialized_model_keys": sorted(allowed_missing),
    }


def validate_checkpoint(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_experiment_config: dict[str, Any] | None = None,
    expected_source_provenance: dict[str, Any] | None = None,
    allow_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Validate a checkpoint without mutating a model or optimizer.

    This is the shared compatibility boundary for resume, evaluation, and
    cloud preflight. Training-only operational fields remain relocatable via
    ``_resume_config_view``, while architecture and behavioral fields must
    match exactly.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    _validate_payload(
        payload,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_experiment_config=expected_experiment_config,
        expected_source_provenance=expected_source_provenance,
        allow_source_mismatch=allow_source_mismatch,
    )
    metadata = _checkpoint_metadata(payload)
    metadata["path"] = str(path)
    return metadata


def load_checkpoint_for_evaluation(
    path: str | Path,
    *,
    model: torch.nn.Module,
    expected_manifest_sha256: str | None = None,
    expected_experiment_config: dict[str, Any] | None = None,
    expected_source_provenance: dict[str, Any] | None = None,
    allow_source_mismatch: bool = False,
) -> dict[str, Any]:
    """Validate and load a full training checkpoint for evaluation.

    Strict state-dict loading only protects tensor structure. The explicit
    experiment-config check protects behavioral settings such as Bank write
    policy, memory visibility, pass schedules, and recurrence configuration.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    _validate_payload(
        payload,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_experiment_config=expected_experiment_config,
        expected_source_provenance=expected_source_provenance,
        allow_source_mismatch=allow_source_mismatch,
    )
    model.load_state_dict(payload["model"], strict=True)
    metadata = {
        "path": str(path),
        "format_version": FORMAT_VERSION,
        "train_state": payload["train_state"],
        "experiment_config": payload["experiment_config"],
        "data_manifest_sha256": payload["data_manifest_sha256"],
        "source_provenance": payload["source_provenance"],
        "checkpoint_metadata": payload["checkpoint_metadata"],
    }
    return metadata


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_manifest_sha256: str,
    expected_experiment_config: dict[str, Any] | None = None,
    expected_source_provenance: dict[str, Any] | None = None,
    allow_source_mismatch: bool = False,
    pass_scheduler=None,
) -> tuple[TrainState, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    _validate_payload(
        payload,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_experiment_config=expected_experiment_config,
        expected_source_provenance=expected_source_provenance,
        allow_source_mismatch=allow_source_mismatch,
        pass_scheduler=pass_scheduler,
    )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if pass_scheduler is not None:
        pass_scheduler.load_state_dict(payload["pass_scheduler"])
    restore_rng_state(payload["rng"])
    return TrainState(**payload["train_state"]), payload["sampler"]


def load_latest_valid_checkpoint(
    run_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_manifest_sha256: str,
    expected_experiment_config: dict[str, Any] | None = None,
    expected_source_provenance: dict[str, Any] | None = None,
    allow_source_mismatch: bool = False,
    pass_scheduler=None,
) -> tuple[Path, TrainState, dict, bool]:
    """Load current generation, falling back to the previous valid one."""
    candidates = candidate_checkpoint_paths(run_dir)
    if not candidates:
        raise FileNotFoundError("no checkpoint generations found")
    errors: list[str] = []
    for index, path in enumerate(candidates):
        if not path.exists():
            errors.append(f"{path.name}: missing")
            continue
        try:
            # Inspect before mutating the live model/optimizer. A truncated
            # newest generation must fall back without partially restoring it.
            inspect_checkpoint(path)
            state, sampler = load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_experiment_config=expected_experiment_config,
                expected_source_provenance=expected_source_provenance,
                allow_source_mismatch=allow_source_mismatch,
                pass_scheduler=pass_scheduler,
            )
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        return path, state, sampler, index > 0
    raise RuntimeError("no valid checkpoint generation: " + "; ".join(errors))
