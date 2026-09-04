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
        json.dumps({"event": "segment_end", "reason": "completed"}) + "\n",
        encoding="utf-8",
    )
    (run / campaign.TRANSFER_MANIFEST).write_text(
        campaign._local_manifest(run), encoding="utf-8"
    )

    assert campaign._local_complete(run)

    pointer = json.loads(
        (run / "checkpoints" / "latest.json").read_text(encoding="utf-8")
    )
    current = run / "checkpoints" / pointer["current"]
    current.write_bytes(b"truncated")
    (run / campaign.TRANSFER_MANIFEST).write_text(
        campaign._local_manifest(run), encoding="utf-8"
    )
    assert not campaign._local_complete(run)


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
