import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_inference_efficiency", ROOT / "scripts/benchmark_inference_efficiency.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_default_configs_match_active_study_arms():
    study = ROOT / "benchmarks/development/frozen_backbone_comparison"
    manifest = yaml.safe_load((study / "STUDY.yaml").read_text())
    expected = {study / arm["config"] for arm in manifest["arms"]}
    assert {ROOT / path for path in benchmark.DEFAULT_CONFIGS} == expected
    assert all(path.is_file() for path in expected)


def test_cost_projection_keeps_two_recurrent_mergers_separate():
    rows = []
    for arm, scale in (("projected", 1), ("recirculation", 10)):
        for operation, mode in (("full_sequence", "k4"), ("prefill", "k1_standard"),
                                ("prefill", "k4_feedback"), ("decode_curve", "feedback_k4"),
                                ("decode_curve", "full_diagnostic_k4")):
            rows.append({
                "arm": arm, "variant": "recurrent_memory", "precision": "fp32",
                "operation": operation, "mode": mode, "prompt_tokens": 1,
                "timing": {"median_ms": 100 * scale},
            })
    settings = dict(variant="recurrent_memory", precision="fp32", sequence_length=2048,
                    validation_tokens=4096, routine_blocks=1)
    projected = benchmark._derived_costs(rows, arm="projected", **settings)
    recurrent = benchmark._derived_costs(rows, arm="recirculation", **settings)
    assert projected["full_k4_validation_seconds"] == pytest.approx(0.2)
    assert recurrent["full_k4_validation_seconds"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="explicit arm"):
        benchmark._derived_costs(rows, **settings)
