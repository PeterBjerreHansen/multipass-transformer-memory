from pathlib import Path

import pytest

from tiny_mistral_mptt.studies import StudyValidationError, verify_study


def _with_data_pin(manifest: str) -> str:
    return manifest.replace(
        "question:",
        "data_artifacts:\n"
        f"  data/dolmino/wiring_2048: {'a' * 64}\n"
        "question:",
        1,
    )


def _write_config(
    path: Path,
    *,
    output_dir: str,
    batch_size: int = 1,
    passes: int = 2,
    init_from: str | None = None,
) -> None:
    lines = [
        "variant: recirculation",
        "phase: B",
        "model_dir: checkpoints/TinyMistral-248M-v3",
        "data_dir: data/dolmino/wiring_2048",
        f"output_dir: {output_dir}",
        "device: cpu",
        "dtype: float32",
        "attention_backend: reference",
        "recirculation_mode: adaptive",
        "recirculation_source_layer: 2",
        "recirculation_destination_layer: 0",
        "recirculation_alpha: 0.1",
        "seed: 1337",
        "architecture_seed: 4242",
        f"batch_size: {batch_size}",
        "grad_accum_steps: 1",
        "max_unique_tokens: 2048",
        "learning_rate: 1.0e-6",
        "pretrained_learning_rate: 1.0e-6",
        "added_learning_rate: 1.0e-6",
        "lr_schedule: {type: constant}",
        "weight_decay: 0.01",
        "grad_clip: 1.0",
        "pass_schedule:",
        "  - probabilities:",
        f"      {passes}: 1.0",
        (
            "pass_loss_weights: [0.25, 0.75]"
            if passes == 2
            else "pass_loss_weights: [0.05, 0.20, 0.75]"
        ),
        "eval_every_tokens: 2048",
        "eval_batches: 1",
        "eval_passes: 8",
        "checkpoint_every_tokens: 2048",
        "resume_from: null",
        f"init_from: {init_from}" if init_from is not None else "init_from: null",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0'\n", encoding="utf-8"
    )
    study = tmp_path / "benchmarks" / "development" / "example"
    study.mkdir(parents=True)
    return study


def test_study_allows_only_declared_experimental_axes(tmp_path):
    study = _repo(tmp_path)
    prefix = "benchmarks/development/example/results"
    _write_config(study / "k2.yaml", output_dir=f"{prefix}/k2", passes=2)
    _write_config(study / "k3.yaml", output_dir=f"{prefix}/k3", passes=3)
    (study / "STUDY.yaml").write_text(
        _with_data_pin("""name: example
status: complete
question: Does K matter?
arms:
  - {id: k2, config: k2.yaml}
  - {id: k3, config: k3.yaml}
comparisons:
  - name: k
    arms: [k2, k3]
    experimental_axes: [pass_schedule, pass_loss_weights]
"""),
        encoding="utf-8",
    )
    result = verify_study(study)
    assert result.arm_ids == ("k2", "k3")


def test_study_rejects_undeclared_execution_difference(tmp_path):
    study = _repo(tmp_path)
    prefix = "benchmarks/development/example/results"
    _write_config(study / "a.yaml", output_dir=f"{prefix}/a", batch_size=1)
    _write_config(study / "b.yaml", output_dir=f"{prefix}/b", batch_size=2)
    (study / "STUDY.yaml").write_text(
        _with_data_pin("""name: example
status: complete
question: Compare arms.
arms:
  - {id: a, config: a.yaml}
  - {id: b, config: b.yaml}
comparisons:
  - name: pair
    arms: [a, b]
    experimental_axes: []
"""),
        encoding="utf-8",
    )
    with pytest.raises(StudyValidationError, match="batch_size"):
        verify_study(study)


def test_study_rejects_orphan_runnable_config(tmp_path):
    study = _repo(tmp_path)
    prefix = "benchmarks/development/example/results"
    _write_config(study / "a.yaml", output_dir=f"{prefix}/a")
    _write_config(study / "orphan.yaml", output_dir=f"{prefix}/orphan")
    (study / "STUDY.yaml").write_text(
        _with_data_pin("""name: example
status: complete
question: Compare arm.
arms:
  - {id: a, config: a.yaml}
comparisons: []
"""),
        encoding="utf-8",
    )
    with pytest.raises(StudyValidationError, match="orphan.yaml"):
        verify_study(study)


def test_study_requires_initialization_match_unless_explicitly_allowed(tmp_path):
    study = _repo(tmp_path)
    prefix = "benchmarks/development/example/results"
    _write_config(study / "a.yaml", output_dir=f"{prefix}/a", init_from="a.pt")
    _write_config(study / "b.yaml", output_dir=f"{prefix}/b", init_from="b.pt")
    manifest = study / "STUDY.yaml"
    manifest.write_text(
        _with_data_pin("""name: example
status: complete
question: Compare initialization handling.
arms:
  - {id: a, config: a.yaml}
  - {id: b, config: b.yaml}
comparisons:
  - name: pair
    arms: [a, b]
    experimental_axes: []
"""),
        encoding="utf-8",
    )
    with pytest.raises(StudyValidationError, match="init_from"):
        verify_study(study)

    manifest.write_text(
        _with_data_pin("""name: example
status: complete
question: Compare initialization handling.
arms:
  - {id: a, config: a.yaml}
  - {id: b, config: b.yaml}
comparisons:
  - name: pair
    arms: [a, b]
    experimental_axes: []
    allowed_differences: [init_from]
"""),
        encoding="utf-8",
    )
    verify_study(study)


def test_core_study_must_be_locked(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0'\n", encoding="utf-8"
    )
    study = tmp_path / "benchmarks" / "core" / "main_comparison"
    study.mkdir(parents=True)
    prefix = "benchmarks/core/main_comparison/results"
    _write_config(study / "arm.yaml", output_dir=f"{prefix}/arm")
    (study / "STUDY.yaml").write_text(
        """name: main_comparison
status: active
question: Is the main claim supported?
arms:
  - {id: arm, config: arm.yaml}
comparisons: []
""",
        encoding="utf-8",
    )
    with pytest.raises(StudyValidationError, match="status=locked"):
        verify_study(study)


def test_study_records_exact_data_manifest_hash(tmp_path):
    study = _repo(tmp_path)
    prefix = "benchmarks/development/example/results"
    _write_config(study / "arm.yaml", output_dir=f"{prefix}/arm")
    digest = "a" * 64
    (study / "STUDY.yaml").write_text(
        f"""name: example
status: planned
question: Is the data identity pinned?
data_artifacts:
  data/dolmino/wiring_2048: {digest}
learning_rates_qualified: false
arms:
  - {{id: arm, config: arm.yaml}}
comparisons: []
""",
        encoding="utf-8",
    )

    result = verify_study(study)

    assert result.data_artifacts == (("data/dolmino/wiring_2048", digest),)
    assert result.learning_rates_qualified is False


def test_study_rejects_unpinned_arm_data_directory(tmp_path):
    study = _repo(tmp_path)
    prefix = "benchmarks/development/example/results"
    _write_config(study / "arm.yaml", output_dir=f"{prefix}/arm")
    (study / "STUDY.yaml").write_text(
        f"""name: example
status: planned
question: Is every arm data input pinned?
data_artifacts:
  data/dolmino/other: {'a' * 64}
arms:
  - {{id: arm, config: arm.yaml}}
comparisons: []
""",
        encoding="utf-8",
    )

    with pytest.raises(StudyValidationError, match="exactly match"):
        verify_study(study)


def test_active_study_requires_data_manifest_pins(tmp_path):
    study = _repo(tmp_path)
    prefix = "benchmarks/development/example/results"
    _write_config(study / "arm.yaml", output_dir=f"{prefix}/arm")
    (study / "STUDY.yaml").write_text(
        """name: example
status: active
question: Is the active data identity pinned?
arms:
  - {id: arm, config: arm.yaml}
comparisons: []
""",
        encoding="utf-8",
    )

    with pytest.raises(StudyValidationError, match="must pin every arm data_dir"):
        verify_study(study)
