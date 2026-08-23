from pathlib import Path

import yaml

from tiny_mistral_mptt.config import ExperimentConfig, load_experiment_config
from tiny_mistral_mptt.data.config import load_data_config
from tiny_mistral_mptt.studies import discover_studies, verify_study

ROOT = Path(__file__).resolve().parents[1]


def _control_configs() -> list[Path]:
    return sorted((ROOT / "benchmarks" / "controls").glob("**/*.yaml"))


def _development_configs() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "benchmarks" / "development").glob("**/*.yaml")
        if path.name != "STUDY.yaml"
    )


def test_default_experiment_config_uses_active_2048_context_and_local_generated_output():
    cfg = ExperimentConfig()
    assert cfg.data_dir == "data/dolmino/wiring_2048"
    assert cfg.output_dir == "benchmarks/controls/smoke/results/generated/vanilla"


def test_data_recipes_live_beside_materialized_artifacts():
    for name in ("wiring_2048", "pilot_2048", "gpu_2048"):
        path = ROOT / "data" / "dolmino" / name / "config.yaml"
        assert path.exists()
        cfg = load_data_config(path)
        assert cfg.output_dir == f"data/dolmino/{name}"
        assert cfg.sequence_length == 2048


def test_evaluation_suites_are_reusable_assets_not_data_recipes():
    suite_dir = ROOT / "evaluation" / "suites"
    for name in ("quick.yaml", "full.yaml"):
        suite = yaml.safe_load((suite_dir / name).read_text(encoding="utf-8"))
        assert suite["tasks"]
        assert all("name" in task and "num_fewshot" in task for task in suite["tasks"])
    assert not (ROOT / "data" / "lm_evaluation").exists()


def test_control_configs_parse_and_write_to_local_generated_results():
    configs = _control_configs()
    assert configs
    for path in configs:
        cfg = load_experiment_config(path)
        control_dir = path.parent
        expected_prefix = (control_dir / "results" / "generated").relative_to(ROOT)
        output = Path(cfg.output_dir)
        assert output.is_relative_to(expected_prefix)


def test_development_studies_verify_semantically():
    manifests = discover_studies(ROOT)
    assert manifests
    for manifest in manifests:
        verify_study(manifest)


def test_active_pipeline_shelves_retired_controls_and_covers_all_tape_policies():
    development = ROOT / "benchmarks" / "development"
    expected_tape_modes = {"dense", "periodic", "memory_token"}

    for stage_name, expected_retention in (
        ("stage_1_wiring", 1),
        ("stage_2_local_smoke", 1),
        ("stage_3_cloud_pilot", 2),
    ):
        stage = development / stage_name
        manifest = yaml.safe_load((stage / "STUDY.yaml").read_text(encoding="utf-8"))
        configs = [
            load_experiment_config(stage / arm["config"])
            for arm in manifest["arms"]
        ]

        assert all(
            cfg.variant not in {"fbt", "memory_add", "tape_add_hybrid"}
            for cfg in configs
        )
        assert {
            cfg.variant for cfg in configs if "hybrid" in cfg.variant
        } == {"tape_recirculation_hybrid"}
        tape_configs = [cfg for cfg in configs if cfg.variant == "tape"]
        assert {cfg.memory_write_mode for cfg in tape_configs} == expected_tape_modes
        explicit = next(
            cfg for cfg in tape_configs if cfg.memory_write_mode == "memory_token"
        )
        assert explicit.memory_write_stride == 32
        assert explicit.memory_token_visibility == "write_only"
        assert all(cfg.checkpoint_keep_last == expected_retention for cfg in configs)

    stage5 = development / "stage_5_cloud_100m"
    manifest = yaml.safe_load((stage5 / "STUDY.yaml").read_text(encoding="utf-8"))
    configs = [
        load_experiment_config(stage5 / arm["config"])
        for arm in manifest["arms"]
    ]
    assert all(
        cfg.variant not in {"fbt", "memory_add", "tape_add_hybrid"}
        for cfg in configs
    )


def test_wiring_and_pilot_each_consume_one_complete_training_split():
    development = ROOT / "benchmarks" / "development"
    cases = (
        ("stage_1_wiring", "wiring_2048", 5_242_880),
        ("stage_3_cloud_pilot", "pilot_2048", 10_485_760),
    )
    for stage_name, artifact_name, expected_tokens in cases:
        data = load_data_config(
            ROOT / "data" / "dolmino" / artifact_name / "config.yaml"
        )
        manifest = yaml.safe_load(
            (development / stage_name / "STUDY.yaml").read_text(encoding="utf-8")
        )
        configs = [
            load_experiment_config(development / stage_name / arm["config"])
            for arm in manifest["arms"]
        ]

        assert data.train_tokens == expected_tokens
        assert all(cfg.data_dir == f"data/dolmino/{artifact_name}" for cfg in configs)
        assert all(cfg.max_unique_tokens == data.train_tokens for cfg in configs)

    pilot = load_data_config(
        ROOT / "data" / "dolmino" / "pilot_2048" / "config.yaml"
    )
    wiring = load_data_config(
        ROOT / "data" / "dolmino" / "wiring_2048" / "config.yaml"
    )
    assert pilot.validation_tokens >= 256 * pilot.sequence_length
    assert pilot.validation_tokens == wiring.validation_tokens
    assert pilot.seed == wiring.seed
    assert pilot.dataset_repo == wiring.dataset_repo
    assert pilot.revision == wiring.revision
    assert pilot.shuffle_buffer == wiring.shuffle_buffer
    assert wiring.train_skip_tokens == 0
    assert pilot.train_skip_tokens == wiring.train_tokens


def test_phase_a_recirculation_defaults_to_1e_minus_4():
    cfg = load_experiment_config(
        ROOT
        / "benchmarks"
        / "development"
        / "stage_1_wiring"
        / "recirculation_adaptive_wiring.yaml"
    )
    assert cfg.learning_rate == 1e-4
    assert cfg.added_lr == 1e-4


def test_runnable_local_configs_keep_one_checkpoint_generation():
    for path in _control_configs() + _development_configs():
        cfg = load_experiment_config(path)
        if cfg.device in {"cpu", "mps"}:
            assert cfg.checkpoint_keep_last == 1, path


def test_active_configs_do_not_depend_on_historical_or_legacy_namespaces():
    configs = _control_configs() + _development_configs()
    assert configs
    for path in configs:
        text = path.read_text(encoding="utf-8")
        assert "benchmarks/historical/" not in text
        assert "experiments/" not in text
        assert "configs/" not in text
        assert "runs/" not in text


def test_training_efficiency_defaults_to_2048_cases():
    suite = yaml.safe_load(
        (ROOT / "benchmarks" / "efficiency" / "suites" / "training.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert {case["sequence_length"] for case in suite["cases"]} == {2048}


def test_root_readme_links_active_pipeline_and_explains_config_locality():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "benchmarks/development/experimental_pipeline.md" in readme
    assert "There is intentionally no central `configs/` directory" in readme
    assert "results/generated/" in readme


def test_deleted_diagnostic_studies_are_not_referenced_by_current_docs():
    current = [ROOT / "README.md", ROOT / "docs", ROOT / "benchmarks" / "development"]
    text_parts = []
    for path in current:
        if path.is_file():
            text_parts.append(path.read_text(encoding="utf-8"))
        else:
            text_parts.extend(
                file.read_text(encoding="utf-8")
                for file in path.rglob("*")
                if file.is_file() and file.suffix in {".md", ".yaml"}
            )
    text = "\n".join(text_parts)
    assert "pass_stability/" not in text
    assert "exact_vs_recurrent_inference/" not in text


def test_gpu_substrate_preserves_validated_2048_token_optimizer_batch():
    cfg = load_experiment_config(ROOT / "benchmarks" / "controls" / "substrate" / "gpu.yaml")
    data = load_data_config(ROOT / "data" / "dolmino" / "gpu_2048" / "config.yaml")
    assert cfg.batch_size == 1
    assert cfg.grad_accum_steps == 1
    assert cfg.batch_size * cfg.grad_accum_steps * data.sequence_length == 2048
    assert cfg.max_unique_tokens == data.train_tokens == 100_007_936
    assert data.train_tokens % (8 * data.sequence_length) == 0
    assert data.validation_tokens > 0
    assert data.train_skip_tokens == 0


def test_ci_runs_canonical_check_gate():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "- run: make check" in workflow
