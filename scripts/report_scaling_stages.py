#!/usr/bin/env python
"""Audit staged study targets against parameter and training-FLOP estimates."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.config import load_data_config
from tiny_mistral_mptt.studies import verify_study


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm_configs(manifest_path: Path) -> dict[str, object]:
    import yaml

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return {
        str(arm["id"]): load_experiment_config(
            manifest_path.parent / str(arm["config"])
        )
        for arm in raw.get("arms", [])
    }


def build_report(study_path: Path) -> dict:
    manifest_path = study_path / "STUDY.yaml" if study_path.is_dir() else study_path
    verification = verify_study(manifest_path)
    if not verification.stages:
        raise ValueError("study has no declared stages")
    configs = _arm_configs(manifest_path)
    model_configs = {Path(config.model_dir) / "config.json" for config in configs.values()}
    if len(model_configs) != 1:
        raise ValueError("staged report requires one common model config")
    model_config_path = model_configs.pop()
    if not model_config_path.is_absolute():
        model_config_path = ROOT / model_config_path

    flops_module = _load_script(
        "estimate_training_flops_for_stages", ROOT / "scripts" / "estimate_training_flops.py"
    )
    wiring_module = _load_script(
        "report_wiring_budgets_for_stages", ROOT / "scripts" / "report_wiring_budgets.py"
    )
    flops_report = flops_module.build_study_report(
        study_path=manifest_path,
        config_path=str(model_config_path),
    )
    flops_by_arm = {row["arm"]: row for row in flops_report["results"]}

    baseline_ids = [
        arm_id
        for arm_id, row in flops_by_arm.items()
        if abs(float(row["relative_training_flops"]) - 1.0) < 1e-12
    ]
    if len(baseline_ids) != 1:
        raise ValueError("staged report requires exactly one 1.0x FLOP baseline")
    baseline_id = baseline_ids[0]
    baseline_per_token = float(
        flops_by_arm[baseline_id]["estimated_training_flops_per_token_presentation"]
    )
    baseline_cfg = configs[baseline_id]
    data_path = Path(baseline_cfg.data_dir)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    sequence_length = load_data_config(data_path / "config.yaml").sequence_length
    baseline_quantum = baseline_cfg.batch_size * baseline_cfg.grad_accum_steps * sequence_length

    stages = []
    for stage in verification.stages:
        targets = dict(stage.targets)
        arm_rows = []
        for arm_id, target in targets.items():
            per_token = float(
                flops_by_arm[arm_id]["estimated_training_flops_per_token_presentation"]
            )
            exact_baseline_tokens = target * per_token / baseline_per_token
            nearest_baseline_tokens = round(exact_baseline_tokens / baseline_quantum) * baseline_quantum
            arm_rows.append(
                {
                    "arm": arm_id,
                    "target_token_presentations": target,
                    "estimated_training_flops": target * per_token,
                    "exact_baseline_token_equivalent": exact_baseline_tokens,
                    "nearest_baseline_optimizer_batch_tokens": nearest_baseline_tokens,
                }
            )
        feedback_rows = [row for row in arm_rows if row["arm"] != baseline_id]
        most_expensive = max(feedback_rows, key=lambda row: row["estimated_training_flops"])
        baseline_row = next(row for row in arm_rows if row["arm"] == baseline_id)
        stages.append(
            {
                "name": stage.name,
                "targets": targets,
                "arms": arm_rows,
                "most_expensive_feedback_arm": most_expensive["arm"],
                "baseline_compute_coverage": (
                    baseline_row["estimated_training_flops"]
                    / most_expensive["estimated_training_flops"]
                ),
            }
        )

    wiring = wiring_module.build_report(
        manifest_path, sequence_length=sequence_length
    )
    return {
        "schema_version": 1,
        "study": verification.name,
        "baseline_arm": baseline_id,
        "baseline_optimizer_batch_tokens": baseline_quantum,
        "stages": stages,
        "parameters": {
            "matched_groups": wiring["matched_groups"],
            "arms": [
                {
                    key: row[key]
                    for key in (
                        "arm",
                        "variant",
                        "added_parameters",
                        "total_parameters",
                        "site_count",
                    )
                }
                for row in wiring["arms"]
            ],
        },
        "flops": flops_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.study)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        for stage in report["stages"]:
            print(
                f"{stage['name']}: baseline compute coverage="
                f"{stage['baseline_compute_coverage']:.6f}"
            )


if __name__ == "__main__":
    main()
