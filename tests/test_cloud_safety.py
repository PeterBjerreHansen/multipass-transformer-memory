from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tiny_mistral_mptt.training.checkpoint import TrainState, save_checkpoint_generation


ROOT = Path(__file__).resolve().parents[1]


def _load_extensionless(name: str, filename: str):
    path = ROOT / "scripts" / filename
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def _objects():
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    return model, optimizer


def test_campaign_requires_verified_manifest_and_readable_current_checkpoint(tmp_path):
    campaign = _load_extensionless("run_cloud_study_test", "run-cloud-study")
    run = tmp_path / "arm"
    model, optimizer = _objects()
    save_checkpoint_generation(
        run,
        model=model,
        optimizer=optimizer,
        sampler_state={"position": 8},
        train_state=TrainState(unique_tokens_seen=8, model_positions_seen=8),
        experiment_config={"variant": "vanilla"},
        data_manifest_sha256="manifest",
        keep_last=1,
    )
    (run / "run.json").write_text("{}\n", encoding="utf-8")
    (run / "segments.jsonl").write_text(
        json.dumps(
            {
                "event": "segment_end",
                "reason": "completed",
                "end_unique_tokens": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run / campaign.TRANSFER_MANIFEST).write_text(
        campaign._local_manifest(run), encoding="utf-8"
    )

    assert campaign._local_complete(run)
    assert campaign._local_complete(run, required_tokens=8)
    assert not campaign._local_complete(run, required_tokens=9)

    pointer = json.loads(
        (run / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    current = run / "checkpoints" / pointer["current"]
    current.write_bytes(b"truncated")
    (run / campaign.TRANSFER_MANIFEST).write_text(
        campaign._local_manifest(run), encoding="utf-8"
    )
    assert not campaign._local_complete(run)


def test_intermediate_cloud_stage_retains_remote_checkpoint(tmp_path):
    campaign = _load_extensionless("run_cloud_study_stage_test", "run-cloud-study")
    args = SimpleNamespace(
        study_dir="benchmarks/development/study",
        local_root=tmp_path,
        host="example.invalid",
        vm_id="vm-1",
        remote_root="/workspace/repo",
        ssh_key=tmp_path / "key",
        verda="verda",
        poll_seconds=60.0,
        transfer="all",
        no_notify=True,
    )

    command = campaign._controller_command(
        args,
        "arm",
        "arm.yaml",
        "a" * 64,
        until_unique_tokens=100,
        final_stage=False,
    )

    assert command[-3:-1] == ["--expected-data-manifest-sha256", "a" * 64]
    assert "--until-unique-tokens" in command
    assert "--delete-remote-output" not in command


def test_locked_cloud_study_defaults_to_metadata_transfer():
    campaign = _load_extensionless("run_cloud_study_transfer_default_test", "run-cloud-study")
    args = campaign._parser().parse_args(
        ["--host", "example.invalid", "--vm-id", "vm-1", "--study-dir", "study"]
    )
    assert args.transfer == "metadata"


def test_completed_early_segment_is_resumable_for_a_later_target(tmp_path):
    controller = _load_extensionless("start_and_watch_stage_resume_test", "start-and-watch")
    output = tmp_path / "arm"
    output.mkdir()
    (output / "segments.jsonl").write_text(
        json.dumps(
            {
                "event": "segment_end",
                "reason": "completed",
                "end_unique_tokens": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-c", controller._status_code(), str(output), "200"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["state"] == "resumable"


def test_remote_identity_helpers_reject_config_or_run_path_mismatch(tmp_path):
    controller = _load_extensionless("start_and_watch_test", "start-and-watch")
    config = tmp_path / "arm.yaml"
    output = tmp_path / "results" / "arm"
    output.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = data_dir / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config.write_text(
        "output_dir: results/arm\ndata_dir: data\n", encoding="utf-8"
    )
    (output / "run.json").write_text(
        json.dumps({"config": {"output_dir": "results/arm"}}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            controller._config_output_code(),
            str(tmp_path),
            str(config),
            str(output),
            manifest_sha256,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            controller._remote_run_identity_code(),
            str(tmp_path),
            str(config),
            str(output),
        ],
        check=True,
    )

    config.write_text(
        "output_dir: results/other\ndata_dir: data\n", encoding="utf-8"
    )
    failed = subprocess.run(
        [
            sys.executable,
            "-c",
            controller._config_output_code(),
            str(tmp_path),
            str(config),
            str(output),
            manifest_sha256,
        ],
        check=False,
    )
    assert failed.returncode != 0


def test_remote_config_helper_rejects_unpinned_data_manifest(tmp_path):
    controller = _load_extensionless("start_and_watch_hash_test", "start-and-watch")
    output = tmp_path / "results" / "arm"
    output.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    config = tmp_path / "arm.yaml"
    config.write_text(
        "output_dir: results/arm\ndata_dir: data\n", encoding="utf-8"
    )

    failed = subprocess.run(
        [
            sys.executable,
            "-c",
            controller._config_output_code(),
            str(tmp_path),
            str(config),
            str(output),
            "a" * 64,
        ],
        check=False,
    )

    assert failed.returncode != 0


def test_remote_helpers_resolve_inherited_config_fields(tmp_path):
    controller = _load_extensionless("start_and_watch_extends_test", "start-and-watch")
    output = tmp_path / "results" / "arm"
    output.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = data_dir / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    (tmp_path / "pilot_base.yml").write_text(
        "output_dir: results/arm\ndata_dir: data\n", encoding="utf-8"
    )
    config = tmp_path / "arm.yaml"
    config.write_text("extends: pilot_base.yml\n", encoding="utf-8")
    (output / "run.json").write_text(
        json.dumps({"config": {"output_dir": "results/arm"}}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-c",
            controller._config_output_code(),
            str(tmp_path),
            str(config),
            str(output),
            manifest_sha256,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            controller._remote_run_identity_code(),
            str(tmp_path),
            str(config),
            str(output),
        ],
        check=True,
    )


def test_remote_project_commands_use_the_synced_environment():
    controller = _load_extensionless("start_and_watch_python_test", "start-and-watch")
    assert controller.REMOTE_PYTHON == ".venv/bin/python"
    source = (ROOT / "scripts" / "start-and-watch").read_text(encoding="utf-8")
    assert "/root/.local/bin" not in source
    assert '"--delete-delay"' in source


def test_cloud_study_blocks_unqualified_learning_rates(monkeypatch, tmp_path):
    campaign = _load_extensionless("run_cloud_study_gate_test", "run-cloud-study")
    study = tmp_path / "benchmarks" / "development" / "comparison"
    study.mkdir(parents=True)
    (study / "STUDY.yaml").write_text(
        "arms:\n  - {id: arm, config: arm.yaml}\n", encoding="utf-8"
    )
    monkeypatch.setattr(campaign, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        campaign,
        "verify_study",
        lambda path: SimpleNamespace(
            status="locked",
            learning_rates_qualified=False,
            data_artifacts=(("data/dolmino/gpu_2048", "a" * 64),),
        ),
    )

    with pytest.raises(SystemExit, match="learning-rate qualification"):
        campaign._study_plan("benchmarks/development/comparison")


def test_cloud_study_plan_resolves_inherited_config_data_dir(monkeypatch, tmp_path):
    campaign = _load_extensionless("run_cloud_study_extends_test", "run-cloud-study")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='test-repo'\nversion='0'\n", encoding="utf-8"
    )
    study = tmp_path / "benchmarks" / "development" / "comparison"
    study.mkdir(parents=True)
    data = tmp_path / "data" / "dolmino" / "gpu_2048"
    data.mkdir(parents=True)
    (data / "manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "base.yml").write_text(
        "\n".join(
            [
                "variant: recurrent_memory",
                "phase: A",
                "model_dir: checkpoints/TinyMistral-248M-v3",
                "data_dir: data/dolmino/gpu_2048",
                "device: cuda",
                "dtype: float32",
                "autocast_dtype: bfloat16",
                "attention_backend: auto",
                "batch_size: 1",
                "grad_accum_steps: 1",
                "max_unique_tokens: 65536",
                "learning_rate: 1.0e-6",
                "added_learning_rate: 1.0e-3",
                "lr_schedule: {type: constant}",
                "weight_decay: 0.01",
                "grad_clip: 1.0",
                "pass_schedule:",
                "  - probabilities: {2: 1.0}",
                "ntp_pass_loss_weights_by_k:",
                "  2: [0.0, 1.0]",
                "memory_window: 1",
                "memory_layers: [3]",
                "recurrent_merger: projected_residual",
                "eval_every_tokens: 65536",
                "eval_batches: 1",
                "eval_passes: 2",
                "checkpoint_every_tokens: 65536",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (study / "arm.yaml").write_text(
        "extends: ../../../base.yml\n"
        "output_dir: benchmarks/development/comparison/results/arm\n",
        encoding="utf-8",
    )
    (study / "STUDY.yaml").write_text(
        "name: comparison\n"
        "status: locked\n"
        "learning_rates_qualified: true\n"
        "question: Does inherited configuration execute?\n"
        "data_artifacts:\n"
        f"  data/dolmino/gpu_2048: {'a' * 64}\n"
        "arms:\n"
        "  - {id: arm, config: arm.yaml}\n"
        "comparisons: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(campaign, "_repo_root", lambda: tmp_path)

    plan = campaign._study_plan("benchmarks/development/comparison")

    assert plan["arm"].config == "arm.yaml"
    assert plan["arm"].until_unique_tokens == 65_536
    assert plan["arm"].final_stage is True


@pytest.mark.parametrize("final_stage", [False, True])
def test_cloud_campaign_retains_vm_until_final_stage(tmp_path, monkeypatch, final_stage):
    campaign = _load_extensionless("run_cloud_study_lifecycle_test", "run-cloud-study")
    key = tmp_path / "key"
    key.touch()
    monkeypatch.setattr(sys, "argv", [
        "run-cloud-study", "--host", "example.invalid", "--vm-id", "vm-1",
        "--study-dir", "benchmarks/development/example",
        "--ssh-key", str(key), "--local-root", str(tmp_path / "results"),
        "--lock-file", str(tmp_path / "lock"), "--no-notify",
    ])
    monkeypatch.setattr(campaign, "_study_plan", lambda *a, **kw: {
        "arm": campaign.CloudStudyArm("arm.yaml", "a" * 64, 100, final_stage),
    })
    completed = iter([False, True])
    monkeypatch.setattr(campaign, "_local_complete", lambda *a, **kw: next(completed))
    commands = []
    monkeypatch.setattr(campaign, "_run", lambda command: commands.append(command))
    deleted = []
    monkeypatch.setattr(campaign, "_delete_vm", lambda args: deleted.append(args.vm_id))

    assert campaign.main() == 0
    assert len(commands) == 1
    assert ("--delete-remote-output" in commands[0]) is final_stage
    assert deleted == (["vm-1"] if final_stage else [])
