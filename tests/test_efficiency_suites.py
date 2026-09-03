from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "benchmarks" / "efficiency" / "suites"


def _suite(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_efficiency_suites_are_explicit_and_non_scientific():
    expected = {
        "training.yaml",
        "context_scaling.yaml",
        "batch_scaling.yaml",
        "precision_mps.yaml",
        "precision_cuda.yaml",
    }
    assert {path.name for path in SUITES.glob("*.yaml")} == expected
    assert (ROOT / "benchmarks" / "efficiency" / "README.md").exists()


def test_efficiency_suite_cases_have_required_dimensions():
    for path in SUITES.glob("*.yaml"):
        raw = _suite(path)
        defaults = raw.get("defaults", {})
        cases = raw.get("cases", [])
        assert cases
        for case in cases:
            merged = {**defaults, **case}
            assert merged["variant"] in {
                "swa_transformer",
                "strided_self_attention",
                "recirculation",
                "memory_attention",
                "dense_and_strided_memory_attention",
            }
            assert merged["passes"] in {1, 2, 3}
            assert merged["sequence_length"] > 0
            assert merged["batch_size"] > 0
            assert merged.get("grad_accum_steps", 1) > 0
            assert merged["parameter_dtype"] == "float32"


def test_shared_scaling_suites_are_device_portable():
    for name in ("training.yaml", "context_scaling.yaml", "batch_scaling.yaml"):
        raw = _suite(SUITES / name)
        assert raw["defaults"]["device"] == "auto"
        assert raw["defaults"]["autocast_dtype"] is None


def test_precision_suites_compare_fp32_and_bfloat16_on_each_backend():
    for backend in ("mps", "cuda"):
        raw = _suite(SUITES / f"precision_{backend}.yaml")
        assert raw["defaults"]["device"] == backend
        modes = {case.get("autocast_dtype") for case in raw["cases"]}
        assert modes == {None, "bfloat16"}
        pairs = {(case["variant"], case["passes"]) for case in raw["cases"]}
    assert pairs == {("swa_transformer", 1), ("recirculation", 2), ("memory_attention", 3)}
