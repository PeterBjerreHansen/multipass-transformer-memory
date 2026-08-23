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


ARCHITECTURE_FIELDS = (
    "variant",
    "sequence_length",
    "memory_window",
    "memory_write_mode",
    "memory_write_stride",
    "memory_token_visibility",
    "memory_layers",
    "memory_position_encoding",
    "recirculation_mode",
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


def _estimate_case(config: MistralConfig, case: dict[str, Any], schedule: dict[int, float]):
    variant = str(case["variant"])
    if variant == "vanilla":
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
        memory_write_mode=case.get("memory_write_mode"),
        memory_write_stride=case.get("memory_write_stride"),
        memory_token_visibility=str(case.get("memory_token_visibility", "visible")),
        memory_layers=memory_layers,
        recirculation_mode=str(case.get("recirculation_mode", "fixed")),
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
        expected_passes = {1} if first["variant"] == "vanilla" else set(schedule)
        if not expected_passes.issubset(observed_passes):
            raise ValueError(
                f"suite architecture {first['variant']!r} lacks required pass rows: "
                f"need {sorted(expected_passes)}, found {sorted(observed_passes)}"
            )
        estimate = _estimate_case(config, first, schedule)
        row = {
            "variant": first["variant"],
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
        "estimator": {
            "flops_per_matmul": 2,
            "backward_multiplier": 3,
            "schedule": {str(key): value for key, value in sorted(schedule.items())},
            "model_config": config.to_dict(),
            "scope": "dominant dense/matmul FLOPs; excludes elementwise, masking, and optimizer arithmetic",
        },
        "suite": str(suite_path),
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmarks/efficiency/suites/stage_5_architectures.yaml"),
    )
    parser.add_argument("--model-config", help="Optional model config.json; defaults to TinyMistral-248M-v3 fields")
    parser.add_argument("--schedule", default="2:0.9,3:0.1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(
        suite_path=args.suite,
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
            print(
                f"{row['variant']}: "
                f"{row['relative_training_flops']:.4f}x relative training FLOPs"
            )


if __name__ == "__main__":
    main()
