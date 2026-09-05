#!/usr/bin/env python
"""Report instantiated added parameters and estimated FLOPs for a wiring study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tiny_mistral.config import MistralConfig
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.flops import estimate_schedule, memory_write_positions
from tiny_mistral_mptt.model_factory import build_variant_from_config
from tiny_mistral_mptt.studies import verify_study


def _initialization(cfg) -> str:
    if cfg.variant == "no_memory_adapter" or cfg.recurrent_merger == "projected_residual":
        return "zero_output_projection"
    if cfg.recurrent_merger == "recirculation":
        return "fixed_alpha_beta_controller"
    return "zero_output_projection"


def _retention(cfg, sequence_length: int) -> dict:
    if cfg.memory_pattern == "strided":
        _, writes, _ = memory_write_positions(
            linguistic_length=sequence_length,
            memory_write_mode=str(cfg.memory_write_mode),
            memory_write_stride=cfg.memory_write_stride,
        )
        return {
            "physical_write_count": len(writes),
            "effective_memory_span_tokens": cfg.memory_window * cfg.memory_write_stride,
        }
    if cfg.memory_pattern == "dense":
        return {
            "physical_write_count": sequence_length,
            "effective_memory_span_tokens": cfg.memory_window,
        }
    if cfg.memory_pattern == "dense_and_strided":
        return {
            "physical_write_count": sequence_length,
            "effective_memory_span_tokens": (
                cfg.memory_dense_window
                + cfg.memory_sparse_window * cfg.memory_sparse_stride
            ),
        }
    return {"physical_write_count": 0, "effective_memory_span_tokens": 0}


def build_report(study_path: Path, *, sequence_length: int) -> dict:
    study = verify_study(study_path)
    arm_configs = list(_study_arm_configs(study_path))
    if tuple(arm_id for arm_id, _, _ in arm_configs) != study.arm_ids:
        raise RuntimeError("verified study arm order changed while building budget report")
    model_dirs = {Path(cfg.model_dir) for _, _, cfg in arm_configs}
    if len(model_dirs) != 1:
        raise RuntimeError("budget report requires one common model_dir across study arms")
    model_config_path = model_dirs.pop() / "config.json"
    model_config = MistralConfig.from_json_file(model_config_path)
    rows = []
    for arm_id, config_path, cfg in arm_configs:
        with torch.device("meta"):
            backbone = MistralForCausalLM(model_config, attention_backend="reference")
            model = build_variant_from_config(cfg, backbone)
        added_parameters = sum(
            parameter.numel() for parameter in model.added_parameters()
        )
        probabilities = cfg.normalized_pass_schedule()[0]["probabilities"]
        estimate = estimate_schedule(
            model_config,
            variant=cfg.variant,
            pass_probabilities=probabilities,
            linguistic_sequence_length=sequence_length,
            memory_window=cfg.memory_window,
            memory_pattern=cfg.memory_pattern,
            memory_write_mode=cfg.memory_write_mode,
            memory_write_stride=cfg.memory_write_stride,
            memory_token_visibility=cfg.memory_token_visibility or "visible",
            memory_layers="all" if cfg.memory_layers is None else cfg.memory_layers,
            memory_num_key_value_heads=cfg.memory_num_key_value_heads,
            memory_dense_window=cfg.memory_dense_window,
            memory_sparse_window=cfg.memory_sparse_window,
            memory_sparse_stride=cfg.memory_sparse_stride,
            recurrent_merger=cfg.recurrent_merger,
            recurrent_controller_hidden_size=cfg.recurrent_controller_hidden_size,
            recurrent_layers=cfg.recurrent_layers,
        )
        site_count = len(cfg.memory_layers) if isinstance(cfg.memory_layers, list) else None
        rows.append({
            "arm": arm_id,
            "variant": cfg.variant,
            "sites": cfg.memory_layers,
            "site_count": site_count,
            "added_parameters": added_parameters,
            "weighted_training_flops_per_sequence": estimate.weighted_training_flops,
            "relative_training_flops": estimate.relative_training_flops,
            "pass_probabilities": probabilities,
            "initialization": _initialization(cfg),
            **_retention(cfg, sequence_length),
        })

    matched_groups = {}
    for site_count in (1, 2):
        group = [
            row for row in rows
            if row["site_count"] == site_count
            and row["arm"] in {
                "no_memory_adapter_one_site_100m",
                "recurrent_projected_residual_multipass_100m",
                "recurrent_recirculation_multipass_100m",
                "dense_memory_attention_one_site_100m",
                "no_memory_adapter_two_site_100m",
                "recurrent_projected_residual_two_site_100m",
                "recurrent_recirculation_two_site_100m",
                "dense_memory_attention_multipass_100m",
            }
        ]
        counts = [row["added_parameters"] for row in group]
        matched_groups[str(site_count)] = {
            "arms": [row["arm"] for row in group],
            "max_to_min_added_parameter_ratio": max(counts) / min(counts),
            "within_ten_percent": max(counts) / min(counts) <= 1.1,
        }
    return {
        "schema_version": 1,
        "study": study.name,
        "sequence_length": sequence_length,
        "model_config": str(model_config_path),
        "matched_groups": matched_groups,
        "arms": rows,
    }


def _study_arm_configs(path: Path):
    """Use the verifier's normalized arm records without widening its API."""
    import yaml
    manifest = path / "STUDY.yaml" if path.is_dir() else path
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    for item in raw["arms"]:
        config_path = manifest.parent / item["config"]
        yield item["id"], config_path, load_experiment_config(config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        default="benchmarks/development/frozen_backbone_comparison/STUDY.yaml",
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = build_report(Path(args.study), sequence_length=args.sequence_length)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
