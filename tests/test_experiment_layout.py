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
        manifest.parent / arm["config"]
        for manifest in discover_studies(ROOT)
        if ROOT / "benchmarks" / "development" in manifest.parents
        for arm in yaml.safe_load(manifest.read_text())["arms"]
    )

def test_default_experiment_config_uses_active_2048_context_and_local_output():
    cfg = ExperimentConfig()
    assert cfg.data_dir == "data/dolmino/wiring_2048"
    assert cfg.output_dir == "benchmarks/controls/smoke/results/vanilla"


def test_data_recipes_live_beside_materialized_artifacts():
    for name in ("wiring_2048", "pilot_2048", "gpu_2048", "gpu_2048_staged"):
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


def test_control_configs_parse_and_write_to_local_results():
    configs = _control_configs()
    assert configs
    for path in configs:
        cfg = load_experiment_config(path)
        control_dir = path.parent
        expected_prefix = (control_dir / "results").relative_to(ROOT)
        output = Path(cfg.output_dir)
        assert output.is_relative_to(expected_prefix)


def test_development_studies_verify_semantically():
    manifests = discover_studies(ROOT)
    assert manifests
    for manifest in manifests:
        verify_study(manifest)


def _study_configs(name: str) -> dict[str, ExperimentConfig]:
    study_dir = ROOT / "benchmarks" / "development" / name
    manifest = yaml.safe_load((study_dir / "STUDY.yaml").read_text(encoding="utf-8"))
    return {
        arm["id"]: load_experiment_config(study_dir / arm["config"])
        for arm in manifest["arms"]
    }


def test_active_study_surface_tracks_manifests_not_diagnostic_directories():
    development = ROOT / "benchmarks" / "development"
    expected = {
        "frozen_backbone_lr_qualification",
        "frozen_backbone_comparison",
    }
    assert {path.parent.name for path in development.rglob("STUDY.yaml")} == expected
    assert {path.parent.name for path in discover_studies(ROOT)} == expected
    assert list((ROOT / "benchmarks" / "core").glob("*/STUDY.yaml")) == []


def test_active_studies_share_2048_data_and_effective_optimizer_batch():
    configs = _development_configs()
    assert configs
    for path in configs:
        cfg = load_experiment_config(path)
        assert cfg.data_dir == "data/dolmino/gpu_2048"
        assert cfg.batch_size * cfg.grad_accum_steps == 32
        assert cfg.variant not in {
            "fbt",
            "memory_add",
            "bank_add_hybrid",
            "strided_attention",
            "recirculation_strided_memory_attention",
        }


def test_frozen_backbone_comparison_has_two_recurrent_mergers_and_three_attention_arms():
    configs = _study_configs("frozen_backbone_comparison")
    assert set(configs) == {
        "recurrent_recirculation_multipass_100m",
        "recurrent_projected_residual_multipass_100m",
        "dense_memory_attention_multipass_100m",
        "strided_memory_attention_multipass_100m",
        "multiscale_memory_attention_multipass_100m",
    }
    assert all(cfg.phase == "A" for cfg in configs.values())
    assert {(cfg.batch_size, cfg.grad_accum_steps) for cfg in configs.values()} == {
        (8, 4)
    }
    assert {
        cfg.batch_size * cfg.grad_accum_steps * 2048
        for cfg in configs.values()
    } == {65_536}
    assert {cfg.train_log_every_tokens for cfg in configs.values()} == {327_680}
    assert {cfg.max_unique_tokens for cfg in configs.values()} == {100_007_936}
    assert {tuple(cfg.snapshot_at_tokens) for cfg in configs.values()} == {
        (3_276_800, 5_013_504, 10_027_008, 20_021_248, 50_003_968, 100_007_936)
    }
    multipass = configs
    assert all(
        cfg.normalized_pass_schedule()[0]["probabilities"] == {2: 0.9, 3: 0.1}
        for cfg in multipass.values()
    )
    assert all(
        cfg.ntp_pass_loss_weights_by_k == {2: [0.0, 1.0], 3: [0.0, 0.0, 1.0]}
        for cfg in multipass.values()
    )
    strided = configs["strided_memory_attention_multipass_100m"]
    assert strided.variant == "strided_memory_attention"
    assert strided.memory_write_mode == "strided"
    assert strided.memory_write_stride == 32
    multiscale = configs["multiscale_memory_attention_multipass_100m"]
    assert multiscale.variant == "multiscale_memory_attention"
    assert (multiscale.memory_dense_window, multiscale.memory_sparse_window) == (32, 32)
    assert multiscale.memory_sparse_stride == 32
    assert {cfg.added_learning_rate for cfg in configs.values()} == {3.0e-4}
    recurrent = [cfg for cfg in configs.values() if cfg.variant == "recurrent_memory"]
    assert {cfg.recurrent_merger for cfg in recurrent} == {"recirculation", "projected_residual"}
    assert all(cfg.memory_layers == [3] and cfg.memory_window == 1 for cfg in recurrent)
    assert all(cfg.recirculation_source_layer is None for cfg in recurrent)


def test_frozen_backbone_lr_qualification_uses_2048_sweep():
    configs = _study_configs("frozen_backbone_lr_qualification")
    assert len(configs) == 20
    for merger in ("recirculation", "projected_residual"):
        assert sum(cfg.recurrent_merger == merger for cfg in configs.values()) == 4
    assert {cfg.data_dir for cfg in configs.values()} == {"data/dolmino/gpu_2048"}
    assert {(cfg.batch_size, cfg.grad_accum_steps) for cfg in configs.values()} == {
        (8, 4)
    }
    assert {cfg.added_learning_rate for cfg in configs.values()} == {
        3.0e-5,
        1.0e-4,
        3.0e-4,
        1.0e-3,
    }


def test_historical_studies_are_preserved_but_not_discovered():
    historical = ROOT / "benchmarks" / "historical"
    assert (historical / "staged_pipeline" / "stage_5_cloud_100m").is_dir()
    assert (historical / "staged_pipeline" / "stage_6_long_continuation").is_dir()
    assert (historical / "exploratory" / "fbt").is_dir()
    discovered = set(discover_studies(ROOT))
    assert not any(path.is_relative_to(historical) for path in discovered)


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


def test_root_readme_links_active_contract_and_explains_config_locality():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "benchmarks/README.md" in readme
    assert "benchmarks/historical/staged_pipeline/" in readme
    assert "There is intentionally no central `configs/` directory" in readme
    assert "results/<arm>/" in readme


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


def test_stage6_evaluation_stream_starts_after_the_long_training_range():
    long_run = load_data_config(
        ROOT / "data" / "dolmino" / "gpu_2048_long_2p5b" / "config.yaml"
    )
    evaluation = load_data_config(
        ROOT / "data" / "dolmino" / "stage_6_evaluation_2048" / "config.yaml"
    )
    assert evaluation.validation_skip_tokens == (
        long_run.validation_tokens
        + long_run.train_skip_tokens
        + long_run.train_tokens
    )
    assert evaluation.seed == long_run.seed
    assert evaluation.dataset_repo == long_run.dataset_repo
    assert evaluation.revision == long_run.revision
    assert evaluation.shuffle_buffer == long_run.shuffle_buffer


def test_ci_runs_canonical_check_gate():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "- run: make check" in workflow


def test_1024_study_configs_and_recipe_are_removed():
    assert not (ROOT / "data/dolmino/paper_1024/config.yaml").exists()
    for path in (ROOT / "benchmarks").rglob("*.yaml"):
        assert "paper_1024" not in path.read_text()
    assert not (ROOT / "benchmarks/historical/retired_1024").exists()
    assert not (ROOT / "benchmarks/historical/exploratory/frozen_backbone_comparison").exists()
