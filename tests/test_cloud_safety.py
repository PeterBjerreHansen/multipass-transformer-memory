from __future__ import annotations

import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

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
    config.write_text("output_dir: results/arm\n", encoding="utf-8")
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

    config.write_text("output_dir: results/other\n", encoding="utf-8")
    failed = subprocess.run(
        [
            sys.executable,
            "-c",
            controller._config_output_code(),
            str(tmp_path),
            str(config),
            str(output),
        ],
        check=False,
    )
    assert failed.returncode != 0
