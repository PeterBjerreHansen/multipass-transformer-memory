from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest
import torch


_RUN_STUDY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_study.py"
_SPEC = importlib.util.spec_from_file_location("run_study", _RUN_STUDY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUN_STUDY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_STUDY)
_should_resume_auto = _RUN_STUDY._should_resume_auto
_verify_pinned_data_artifacts = _RUN_STUDY._verify_pinned_data_artifacts
_wire_arm = _RUN_STUDY._wire_arm


def _write_config(path: Path, *, output_dir: str, init_from: str | None) -> None:
    path.write_text(
        "\n".join(
            [
                f"output_dir: {output_dir}",
                f"init_from: {init_from if init_from is not None else 'null'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_init_from_config_starts_fresh_when_output_is_empty(tmp_path):
    config = tmp_path / "phase-b.yaml"
    _write_config(config, output_dir="results/phase-b", init_from="checkpoints/phase-a.pt")

    assert _should_resume_auto(config, root=tmp_path) is False


def test_init_from_config_resumes_when_output_has_trajectory(tmp_path):
    config = tmp_path / "phase-b.yaml"
    _write_config(config, output_dir="results/phase-b", init_from="checkpoints/phase-a.pt")
    output_dir = tmp_path / "results" / "phase-b"
    output_dir.mkdir(parents=True)
    (output_dir / "run.json").write_text("{}\n", encoding="utf-8")

    assert _should_resume_auto(config, root=tmp_path) is True


def test_config_without_init_from_keeps_auto_resume_behavior(tmp_path):
    config = tmp_path / "phase-a.yaml"
    _write_config(config, output_dir="results/phase-a", init_from=None)

    assert _should_resume_auto(config, root=tmp_path) is True


def test_wiring_executes_the_declared_training_forward(monkeypatch, tmp_path):
    observed = {}

    class FakeConfig:
        device = "cpu"
        init_from = None
        data_dir = "unused"
        memory_write_mode = None
        memory_write_stride = None
        phase = "A"
        training_forward = "parallel_multipass"
        autocast_dtype = None
        batch_size = 3

        def normalized_pass_schedule(self):
            return [{"until_tokens": None, "probabilities": {2: 1.0}}]

        def ntp_loss_weights_for_passes(self, passes):
            assert passes == 2
            return [0.0, 1.0]

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def compute_training_loss(self, input_ids, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(loss=self.weight.square() + input_ids.sum() * 0)

    def _recorded_batch(indices, device):
        observed["batch_indices"] = list(indices)
        return torch.ones((len(observed["batch_indices"]), 2), device=device)

    model = FakeModel()
    monkeypatch.setattr(_RUN_STUDY, "load_experiment_config", lambda path: FakeConfig())
    monkeypatch.setattr(_RUN_STUDY, "resolve_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(_RUN_STUDY, "load_variant_from_config", lambda cfg, device: model)
    monkeypatch.setattr(
        _RUN_STUDY,
        "load_packed_dataset_for_experiment",
        lambda *args, **kwargs: SimpleNamespace(batch=_recorded_batch),
    )
    monkeypatch.setattr(_RUN_STUDY, "configure_phase", lambda model, phase: None)

    _wire_arm(tmp_path / "multipass.yaml", wire_device=None)

    assert observed["training_forward"] == "parallel_multipass"
    assert observed["phase"] == "A"
    assert observed["passes"] == 2
    assert observed["loss_weights"] == [0.0, 1.0]
    assert observed["batch_indices"] == [0, 1, 2]
    assert model.weight.grad is not None


def test_study_execution_checks_pinned_manifest_hash(monkeypatch, tmp_path):
    config_path = tmp_path / "arm.yaml"
    _write_config(config_path, output_dir="results/arm", init_from=None)
    data_dir = tmp_path / "data" / "dolmino" / "wiring_2048"
    data_dir.mkdir(parents=True)
    (data_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    expected = "a" * 64
    verification = SimpleNamespace(
        data_artifacts=(("data/dolmino/wiring_2048", expected),)
    )
    monkeypatch.setattr(_RUN_STUDY, "verify_artifact", lambda path: None)
    monkeypatch.setattr(_RUN_STUDY, "file_sha256", lambda path: expected)

    _verify_pinned_data_artifacts(
        verification,
        {"arm": config_path},
        ["arm"],
        root=tmp_path,
    )

    monkeypatch.setattr(_RUN_STUDY, "file_sha256", lambda path: "b" * 64)
    with pytest.raises(RuntimeError, match="data manifest hash mismatch"):
        _verify_pinned_data_artifacts(
            verification,
            {"arm": config_path},
            ["arm"],
            root=tmp_path,
        )
