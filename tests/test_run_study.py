from pathlib import Path
import importlib.util
from types import SimpleNamespace

import torch


_RUN_STUDY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_study.py"
_SPEC = importlib.util.spec_from_file_location("run_study", _RUN_STUDY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUN_STUDY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_STUDY)
_should_resume_auto = _RUN_STUDY._should_resume_auto
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
        training_forward = "recirculation_bptt"
        recirculation_activation_checkpointing = True
        autocast_dtype = None

        def normalized_pass_schedule(self):
            return [{"until_tokens": None, "probabilities": {1: 1.0}}]

        def ntp_loss_weights_for_passes(self, passes):
            assert passes == 1
            return None

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def compute_training_loss(self, input_ids, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(loss=self.weight.square() + input_ids.sum() * 0)

    model = FakeModel()
    monkeypatch.setattr(_RUN_STUDY, "load_experiment_config", lambda path: FakeConfig())
    monkeypatch.setattr(_RUN_STUDY, "resolve_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(_RUN_STUDY, "load_variant_from_config", lambda cfg, device: model)
    monkeypatch.setattr(
        _RUN_STUDY,
        "load_packed_dataset_for_experiment",
        lambda *args, **kwargs: SimpleNamespace(
            batch=lambda indices, device: torch.ones((1, 2), device=device)
        ),
    )
    monkeypatch.setattr(_RUN_STUDY, "configure_phase", lambda model, phase: None)

    _wire_arm(tmp_path / "bptt.yaml", wire_device=None)

    assert observed["training_forward"] == "recirculation_bptt"
    assert observed["phase"] == "A"
    assert observed["passes"] == 1
    assert observed["loss_weights"] is None
    assert observed["activation_checkpointing"] is True
    assert model.weight.grad is not None
