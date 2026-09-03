#!/usr/bin/env python
"""Estimate relative dominant training FLOPs for an efficiency suite."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import yaml

from tiny_mistral.config import MistralConfig, tiny_mistral_248m_config
from tiny_mistral_mptt.flops import estimate_schedule
from tiny_mistral_mptt.config import (
    canonical_variant_name, load_experiment_config, reject_removed_paper_policy,
)
from tiny_mistral_mptt.data.config import load_data_config
from tiny_mistral_mptt.studies import verify_study


ARCHITECTURE_FIELDS = (
    "variant",
    "training_forward",
    "sequence_length",
    "memory_window",
    "memory_pattern",
    "memory_write_mode",
    "memory_write_stride",
    "memory_token_visibility",
    "memory_layers",
    "memory_position_encoding",
    "memory_dense_window",
    "memory_sparse_window",
    "memory_sparse_stride",
    "sparse_attention_stride",
    "sparse_attention_window",
    "sparse_attention_layers",
    "recirculation_mode",
    "recurrent_merger",
    "recurrent_layers",
    "recirculation_source_layer",
    "recirculation_destination_layer",
    "recirculation_alpha",
)


def _load_suite(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("suite must be a YAML mapping")
    defaults = raw.get("defaults", {})
    cases = raw.get("cases", [])
    if not isinstance(defaults, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("suite requires defaults and a non-empty cases list")
    result: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be a mapping")
        merged = {**defaults, **case}
        if "variant" not in merged or "passes" not in merged:
            raise ValueError(f"case {index} requires variant and passes")
        result.append(merged)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def _architecture_key(case: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_freeze(case.get(field)) for field in ARCHITECTURE_FIELDS)


def _parse_schedule(raw: str) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in raw.split(","):
        key, separator, value = item.partition(":")
        if not separator:
            raise ValueError("schedule must look like 2:0.9,3:0.1")
        passes = int(key)
        probability = float(value)
        if passes <= 0 or probability < 0:
            raise ValueError("schedule pass counts must be positive and probabilities non-negative")
        result[passes] = probability
    if not result or sum(result.values()) <= 0:
        raise ValueError("schedule must contain positive probability mass")
    return result


def _config(path: str | None) -> MistralConfig:
    if path is None:
        return tiny_mistral_248m_config()
    return MistralConfig.from_json_file(path)


def _estimator_metadata(config: MistralConfig) -> dict[str, Any]:
    return {
        "flops_per_matmul": 2,
        "backward_multiplier": 3,
        "model_config": config.to_dict(),
        "scope": "dominant dense/matmul FLOPs; excludes elementwise, masking, and optimizer arithmetic",
        "freeze_accounting": "conventional backward multiplier; not adjusted for frozen parameter gradients",
    }


def _estimate_case(config: MistralConfig, case: dict[str, Any], schedule: dict[int, float]):
    reject_removed_paper_policy(case)
    variant = str(case["variant"])
    implementation_variant = canonical_variant_name(variant)
    training_forward = str(case.get("training_forward", "parallel_multipass"))
    if training_forward != "parallel_multipass":
        raise ValueError(f"unknown training_forward {training_forward!r}")
    if implementation_variant in {"vanilla", "strided_self_attention"}:
        probabilities = {1: 1.0}
    else:
        probabilities = schedule
    memory_layers = case.get("memory_layers", "all")
    return estimate_schedule(
        config,
        variant=variant,
        pass_probabilities=probabilities,
        linguistic_sequence_length=int(case.get("sequence_length", 2048)),
        memory_window=int(case.get("memory_window", 32)),
        memory_pattern=case.get("memory_pattern"),
        memory_write_mode=case.get("memory_write_mode"),
        memory_write_stride=case.get("memory_write_stride"),
        memory_token_visibility=str(case.get("memory_token_visibility", "visible")),
        memory_layers=memory_layers,
        memory_dense_window=case.get("memory_dense_window"),
        memory_sparse_window=case.get("memory_sparse_window"),
        memory_sparse_stride=case.get("memory_sparse_stride"),
        sparse_attention_stride=case.get("sparse_attention_stride"),
        sparse_attention_window=case.get("sparse_attention_window"),
        sparse_attention_layers=case.get("sparse_attention_layers", "all"),
        recirculation_mode=str(case.get("recirculation_mode", "fixed")),
        recurrent_merger=case.get("recurrent_merger"),
        recurrent_layers=case.get("recurrent_layers"),
    )


def build_report(
    *,
    suite_path: Path,
    config_path: str | None,
    schedule: dict[int, float],
) -> dict[str, Any]:
    config = _config(config_path)
    cases = _load_suite(suite_path)
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[_architecture_key(case)].append(case)

    rows: list[dict[str, Any]] = []
    for group_cases in grouped.values():
        first = group_cases[0]
        observed_passes = {int(case["passes"]) for case in group_cases}
        expected_passes = (
            {1}
            if (
                canonical_variant_name(str(first["variant"]))
                in {"vanilla", "strided_self_attention"}
            )
            else set(schedule)
        )
        if not expected_passes.issubset(observed_passes):
            raise ValueError(
                f"suite architecture {first['variant']!r} lacks required pass rows: "
                f"need {sorted(expected_passes)}, found {sorted(observed_passes)}"
            )
        estimate = _estimate_case(config, first, schedule)
        row = {
            "variant": first["variant"],
            "training_forward": first.get(
                "training_forward", "parallel_multipass"
            ),
            "sequence_length": int(first.get("sequence_length", 2048)),
            "relative_training_flops": estimate.relative_training_flops,
            "weighted_training_flops": estimate.weighted_training_flops,
            "baseline_training_flops": estimate.baseline_training_flops,
            "parameters": {
                field: first.get(field)
                for field in ARCHITECTURE_FIELDS
                if field != "variant"
            },
            "estimate": estimate.to_dict(),
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["variant"], str(row["parameters"])))
    return {
        "schema_version": 2,
        "estimator": {
            **_estimator_metadata(config),
            "schedule": {str(key): value for key, value in sorted(schedule.items())},
        },
        "suite": str(suite_path),
        "results": rows,
    }


def build_study_report(
    *,
    study_path: Path,
    config_path: str | None,
) -> dict[str, Any]:
    """Estimate every arm directly from its authoritative study config."""
    manifest_path = study_path / "STUDY.yaml" if study_path.is_dir() else study_path
    verification = verify_study(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    model_config = _config(config_path)
    repository = Path(__file__).resolve().parents[1]
    rows: list[dict[str, Any]] = []

    for arm in manifest["arms"]:
        experiment_path = manifest_path.parent / str(arm["config"])
        experiment = load_experiment_config(experiment_path)
        data_path = Path(experiment.data_dir)
        if not data_path.is_absolute():
            data_path = repository / data_path
        data = load_data_config(data_path / "config.yaml")
        schedule_stages = experiment.normalized_pass_schedule()
        if len(schedule_stages) != 1 or schedule_stages[0]["until_tokens"] is not None:
            raise ValueError(
                f"study arm {arm['id']!r} uses a staged pass schedule; "
                "estimate its stages separately"
            )
        schedule = {
            int(passes): float(probability)
            for passes, probability in schedule_stages[0]["probabilities"].items()
        }
        case = {
            **experiment.to_dict(),
            "sequence_length": data.sequence_length,
            "passes": min(schedule),
        }
        estimate = _estimate_case(model_config, case, schedule)
        per_sequence = float(estimate.weighted_training_flops)
        per_token = per_sequence / data.sequence_length
        rows.append(
            {
                "arm": str(arm["id"]),
                "config": str(arm["config"]),
                "variant": experiment.variant,
                "phase": experiment.phase,
                "training_forward": experiment.training_forward,
                "sequence_length": data.sequence_length,
                "batch_size": experiment.batch_size,
                "grad_accum_steps": experiment.grad_accum_steps,
                "optimizer_batch_sequences": (
                    experiment.batch_size * experiment.grad_accum_steps
                ),
                "optimizer_batch_tokens": (
                    experiment.batch_size
                    * experiment.grad_accum_steps
                    * data.sequence_length
                ),
                "max_unique_tokens": experiment.max_unique_tokens,
                "estimated_training_flops_per_sequence": per_sequence,
                "estimated_training_flops_per_unique_token": per_token,
                "estimated_training_flops_total": (
                    per_token * experiment.max_unique_tokens
                ),
                "relative_training_flops": estimate.relative_training_flops,
                "estimate": estimate.to_dict(),
            }
        )

    return {
        "estimator": _estimator_metadata(model_config),
        "study": str(manifest_path),
        "study_name": verification.name,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--suite",
        type=Path,
    )
    source.add_argument(
        "--study",
        type=Path,
        help="estimate arms directly from a STUDY.yaml or its directory",
    )
    parser.add_argument("--model-config", help="Optional model config.json; defaults to TinyMistral-248M-v3 fields")
    parser.add_argument("--schedule", default="2:1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.study is not None:
        report = build_study_report(
            study_path=args.study,
            config_path=args.model_config,
        )
    else:
        report = build_report(
            suite_path=(
                args.suite
                or Path("benchmarks/efficiency/suites/training.yaml")
            ),
            config_path=args.model_config,
            schedule=_parse_schedule(args.schedule),
        )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        for row in report["results"]:
            label = row.get("arm", row["variant"])
            print(
                f"{label}: "
                f"{row['relative_training_flops']:.4f}x relative training FLOPs"
            )


if __name__ == "__main__":
    main()
