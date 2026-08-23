"""Validation for colocated benchmark study manifests.

A ``STUDY.yaml`` records the scientific question and which runnable configs are
part of the study. Execution settings remain authoritative in those configs;
the manifest deliberately does not duplicate learning rates, token budgets, or
other trajectory details.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml

from .config import ExperimentConfig, load_experiment_config


_ARM_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ALLOWED_STATUS = {"planned", "active", "complete", "locked"}
_ALWAYS_LOCAL_FIELDS = {"output_dir"}
_CONFIG_FIELD_ALIASES = {
    "pass_loss_weights": "ntp_pass_loss_weights",
    "pass_loss_weights_by_k": "ntp_pass_loss_weights_by_k",
}


class StudyValidationError(ValueError):
    """Raised when a study manifest and its runnable configs disagree."""


@dataclass(frozen=True, slots=True)
class StudyArm:
    id: str
    config_path: Path
    config: ExperimentConfig


@dataclass(frozen=True, slots=True)
class StudyVerification:
    manifest_path: Path
    name: str
    status: str
    arm_ids: tuple[str, ...]
    comparison_names: tuple[str, ...]


def _repo_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise StudyValidationError(f"could not locate repository root from {path}")


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StudyValidationError(f"{label} must be a YAML mapping")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise StudyValidationError(f"{label} must be a YAML list")
    return value


def _resolve_inside(base: Path, relative: str, *, label: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise StudyValidationError(f"{label} must be relative to the study directory")
    resolved = (base / path).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise StudyValidationError(f"{label} escapes the study directory: {relative}") from exc
    return resolved


def _comparison_view(
    config: ExperimentConfig,
    *,
    experimental_axes: set[str],
    allowed_differences: set[str],
) -> dict[str, Any]:
    # Study manifests written before the explicit NTP/NMP naming split may
    # still declare the generic pass-loss names. Compare against the canonical
    # serialized config while retaining compatibility with those manifests.
    declared = {
        _CONFIG_FIELD_ALIASES.get(field, field)
        for field in experimental_axes | allowed_differences
    }
    unknown = declared - set(config.__dataclass_fields__)
    if unknown:
        raise StudyValidationError(
            f"unknown comparison config fields: {sorted(unknown)}"
        )
    excluded = _ALWAYS_LOCAL_FIELDS | declared
    return {
        key: value
        for key, value in config.to_dict().items()
        if key not in excluded
    }


def verify_study(path: str | Path) -> StudyVerification:
    """Validate one study manifest and return a compact verification summary."""
    supplied = Path(path)
    manifest_path = supplied / "STUDY.yaml" if supplied.is_dir() else supplied
    if not manifest_path.is_file():
        raise StudyValidationError(f"study manifest does not exist: {manifest_path}")

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    raw = _mapping(raw, label="study manifest")
    unknown_top = sorted(
        set(raw) - {"name", "status", "question", "arms", "comparisons"}
    )
    if unknown_top:
        raise StudyValidationError(f"unknown STUDY.yaml fields: {unknown_top}")

    name = str(raw.get("name", "")).strip()
    status = str(raw.get("status", "")).strip()
    question = str(raw.get("question", "")).strip()
    if not name:
        raise StudyValidationError("study name must be non-empty")
    if status not in _ALLOWED_STATUS:
        raise StudyValidationError(
            f"study status must be one of {sorted(_ALLOWED_STATUS)}; got {status!r}"
        )
    if not question:
        raise StudyValidationError("study question must be non-empty")

    study_dir = manifest_path.parent.resolve()
    repo_root = _repo_root(study_dir)
    if name != study_dir.name:
        raise StudyValidationError(
            f"study name {name!r} must match directory name {study_dir.name!r}"
        )
    relative_study = study_dir.relative_to(repo_root)
    if (
        len(relative_study.parts) >= 2
        and relative_study.parts[:2] == ("benchmarks", "core")
        and status != "locked"
    ):
        raise StudyValidationError("core studies must use status=locked before execution")
    expected_results_root = (study_dir / "results").relative_to(repo_root).as_posix()

    arms_raw = _sequence(raw.get("arms", []), label="arms")
    if status != "planned" and not arms_raw:
        raise StudyValidationError(f"{status} study must declare at least one arm")

    arms: dict[str, StudyArm] = {}
    declared_config_paths: set[Path] = set()
    for index, item in enumerate(arms_raw):
        arm_raw = _mapping(item, label=f"arms[{index}]")
        unknown_arm = sorted(set(arm_raw) - {"id", "config"})
        if unknown_arm:
            raise StudyValidationError(f"unknown arm fields: {unknown_arm}")
        arm_id = str(arm_raw.get("id", "")).strip()
        config_ref = str(arm_raw.get("config", "")).strip()
        if not _ARM_ID.fullmatch(arm_id):
            raise StudyValidationError(f"invalid arm id {arm_id!r}")
        if arm_id in arms:
            raise StudyValidationError(f"duplicate arm id {arm_id!r}")
        if not config_ref:
            raise StudyValidationError(f"arm {arm_id!r} is missing config")
        config_path = _resolve_inside(
            study_dir, config_ref, label=f"arm {arm_id} config"
        )
        if config_path.name == "STUDY.yaml" or not config_path.is_file():
            raise StudyValidationError(f"arm config does not exist: {config_path}")
        config = load_experiment_config(config_path)
        expected_output = f"{expected_results_root}/{arm_id}"
        if Path(config.output_dir).as_posix() != expected_output:
            raise StudyValidationError(
                f"arm {arm_id!r} output_dir must be {expected_output!r}; "
                f"got {config.output_dir!r}"
            )
        arms[arm_id] = StudyArm(arm_id, config_path, config)
        declared_config_paths.add(config_path.resolve())

    # Runnable configs should not silently exist outside the manifest.
    yaml_files = {
        candidate.resolve()
        for candidate in study_dir.glob("*.yaml")
        if candidate.name != "STUDY.yaml"
    }
    orphaned = sorted(path.name for path in yaml_files - declared_config_paths)
    if orphaned:
        raise StudyValidationError(f"runnable configs missing from STUDY.yaml: {orphaned}")

    comparisons_raw = _sequence(raw.get("comparisons", []), label="comparisons")
    comparison_names: list[str] = []
    seen_comparisons: set[str] = set()
    for index, item in enumerate(comparisons_raw):
        comparison = _mapping(item, label=f"comparisons[{index}]")
        unknown_comparison = sorted(
            set(comparison)
            - {"name", "arms", "experimental_axes", "allowed_differences"}
        )
        if unknown_comparison:
            raise StudyValidationError(
                f"unknown comparison fields: {unknown_comparison}"
            )
        comparison_name = str(comparison.get("name", "")).strip()
        if not comparison_name or comparison_name in seen_comparisons:
            raise StudyValidationError(
                f"comparison name must be non-empty and unique: {comparison_name!r}"
            )
        seen_comparisons.add(comparison_name)
        comparison_names.append(comparison_name)
        member_ids = [
            str(value)
            for value in _sequence(
                comparison.get("arms", []),
                label=f"comparison {comparison_name} arms",
            )
        ]
        if len(member_ids) != len(set(member_ids)):
            raise StudyValidationError(
                f"comparison {comparison_name!r} contains duplicate arms"
            )
        if len(member_ids) < 2:
            raise StudyValidationError(
                f"comparison {comparison_name!r} must contain at least two arms"
            )
        missing = [arm_id for arm_id in member_ids if arm_id not in arms]
        if missing:
            raise StudyValidationError(
                f"comparison {comparison_name!r} references unknown arms: {missing}"
            )
        axes = {
            str(value)
            for value in _sequence(
                comparison.get("experimental_axes", []),
                label=f"comparison {comparison_name} experimental_axes",
            )
        }
        allowed_differences = {
            str(value)
            for value in _sequence(
                comparison.get("allowed_differences", []),
                label=f"comparison {comparison_name} allowed_differences",
            )
        }
        overlap = axes & allowed_differences
        if overlap:
            raise StudyValidationError(
                f"comparison {comparison_name!r} lists fields in both experimental_axes "
                f"and allowed_differences: {sorted(overlap)}"
            )
        reference_id = member_ids[0]
        reference = _comparison_view(
            arms[reference_id].config,
            experimental_axes=axes,
            allowed_differences=allowed_differences,
        )
        for arm_id in member_ids[1:]:
            candidate = _comparison_view(
                arms[arm_id].config,
                experimental_axes=axes,
                allowed_differences=allowed_differences,
            )
            differing = sorted(
                key
                for key in set(reference) | set(candidate)
                if reference.get(key) != candidate.get(key)
            )
            if differing:
                raise StudyValidationError(
                    f"comparison {comparison_name!r} differs outside declared axes "
                    f"between {reference_id!r} and {arm_id!r}: {differing}"
                )

    return StudyVerification(
        manifest_path=manifest_path,
        name=name,
        status=status,
        arm_ids=tuple(arms),
        comparison_names=tuple(comparison_names),
    )


def discover_studies(root: str | Path) -> list[Path]:
    """Return active study manifests in deterministic order."""
    root_path = Path(root)
    return sorted(
        path
        for parent in (
            root_path / "benchmarks" / "development",
            root_path / "benchmarks" / "core",
        )
        if parent.exists()
        for path in parent.rglob("STUDY.yaml")
    )
