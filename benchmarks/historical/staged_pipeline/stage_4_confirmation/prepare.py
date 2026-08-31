#!/usr/bin/env python
"""Materialize Stage-4 configs after selecting fast and Memory Attention pilot arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


SEEDS = (2027, 4099)
PILOT_CONFIGS = {
    "swa_transformer": "vanilla_seed1337.yaml",
    "recirculation_adaptive": "recirculation_adaptive_seed1337.yaml",
    "dense_memory_attention": "bank_dense_seed1337.yaml",
    "strided_memory_attention32": "bank_periodic32_seed1337.yaml",
    "memory_token_attention32": "bank_memory_token32_seed1337.yaml",
    "recirculation_strided_memory_attention": "hybrid_recirculation_seed1337.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast",
        required=True,
        choices=("recirculation_adaptive",),
    )
    parser.add_argument(
        "--memory-attention",
        "--bank",
        dest="memory_attention",
        required=True,
        choices=("dense", "strided32", "memory_token32"),
    )
    parser.add_argument(
        "--hybrid",
        choices=("recirculation",),
        default="recirculation",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stage_dir = Path(__file__).resolve().parent
    root = stage_dir.parents[2]
    pilot_dir = root / "benchmarks" / "development" / "stage_3_cloud_pilot"
    memory_key = {
        "dense": "dense_memory_attention",
        "strided32": "strided_memory_attention32",
        "memory_token32": "memory_token_attention32",
    }[args.memory_attention]
    hybrid = "recirculation_strided_memory_attention"
    selected = ("swa_transformer", args.fast, memory_key, hybrid)
    generated: list[tuple[str, str]] = []

    destinations = [stage_dir / "STUDY.yaml"]
    destinations.extend(
        stage_dir / f"{architecture}_seed{seed}.yaml"
        for architecture in selected
        for seed in SEEDS
    )
    existing = [path.name for path in destinations[1:] if path.exists()]
    if destinations[0].exists():
        current_manifest = yaml.safe_load(
            destinations[0].read_text(encoding="utf-8")
        ) or {}
        if current_manifest.get("arms"):
            existing.append(destinations[0].name)
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite an existing confirmation selection: "
            + ", ".join(existing)
        )

    for architecture in selected:
        source = pilot_dir / PILOT_CONFIGS[architecture]
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        for seed in SEEDS:
            arm_id = f"{architecture}_seed{seed}"
            config_name = f"{arm_id}.yaml"
            config = dict(raw)
            config["seed"] = seed
            config["output_dir"] = (
                "benchmarks/development/stage_4_confirmation/"
                f"results/{arm_id}"
            )
            (stage_dir / config_name).write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            generated.append((arm_id, config_name))

    manifest = {
        "name": "stage_4_confirmation",
        "status": "planned",
        "question": (
            "Do the selected Memory Attention and Hybrid replicate against the selected fast-memory "
            "baseline across two additional Phase-B seeds?"
        ),
        "arms": [
            {"id": arm_id, "config": config_name}
            for arm_id, config_name in generated
        ],
        "comparisons": [],
    }
    (stage_dir / "STUDY.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    print(
        f"PASS: prepared Stage 4 fast={args.fast} memory_attention={args.memory_attention} hybrid={args.hybrid} "
        f"arms={','.join(arm_id for arm_id, _ in generated)}"
    )


if __name__ == "__main__":
    main()
