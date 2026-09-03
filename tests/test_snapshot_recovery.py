import json
import shutil

import pytest
import torch

from test_evaluation_contract import make_model
from test_pass_depth_eval import make_artifact
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.training import snapshots
from tiny_mistral_mptt.training.checkpoint import (
    candidate_checkpoint_paths, inspect_checkpoint, load_model_weights,
)
from tiny_mistral_mptt.training.trainer import Trainer


def make_trainer(tmp_path, *, resume=False, **overrides):
    data = tmp_path / "data"
    if not data.exists():
        make_artifact(data)
    cfg = ExperimentConfig.from_dict({
        "variant": "memory_add", "device": "cpu", "model_dir": "unused",
        "data_dir": str(data), "output_dir": str(tmp_path / "run"),
        "max_unique_tokens": 24, "attention_backend": "reference",
        "eval_batches": 0, "eval_every_tokens": 0, "checkpoint_every_tokens": 0,
        "snapshot_at_tokens": [7, 8, 16], "checkpoint_keep_last": 1,
        **overrides,
    })
    return Trainer(model=make_model(), config=cfg,
        train_data=PackedTokenDataset(data, "train"),
        validation_data=PackedTokenDataset(data, "validation"), device=torch.device("cpu"),
        resume_auto=resume, allow_source_mismatch=True)


@pytest.mark.parametrize("failure", ["write", "directory_sync", "sidecar"])
def test_snapshot_restart_completes_without_advancing_model(tmp_path, monkeypatch, failure):
    trainer = make_trainer(tmp_path)
    name = {"write": "save_file", "directory_sync": "fsync_directory", "sidecar": "atomic_write_json"}[failure]
    original = getattr(snapshots, name)
    def fail(*args, **kwargs):
        if failure == "write":
            original(*args, **kwargs)
        raise OSError("interrupted publication")
    monkeypatch.setattr(snapshots, name, fail)
    with pytest.raises(OSError, match="interrupted publication"):
        trainer.train(until_unique_tokens=8)
    checkpoint = candidate_checkpoint_paths(tmp_path / "run")[0]
    assert inspect_checkpoint(checkpoint)["checkpoint_metadata"]["pending_snapshots"] == [7, 8]
    expected = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    path = tmp_path / "run/snapshots/model_000000000008.safetensors"
    if failure != "write":
        # Committed weights are loadable even when the mirror was never written.
        target = make_model()
        load_model_weights(path, model=target, expected_experiment_config=trainer.config.to_dict())
        for key, value in expected.items():
            torch.testing.assert_close(target.state_dict()[key], value, atol=0, rtol=0)
    monkeypatch.setattr(snapshots, name, original)
    resumed = make_trainer(tmp_path, resume=True)
    resumed.train(until_unique_tokens=8)
    assert resumed.state.unique_tokens_seen == 8
    metadata = snapshots.snapshot_metadata(path)
    assert metadata["requested_thresholds"] == [7, 8]
    assert json.loads(path.with_suffix(".json").read_text()) == metadata
    for key, value in expected.items():
        torch.testing.assert_close(resumed.model.state_dict()[key], value, atol=0, rtol=0)
    # Repeating recovery from the same durable checkpoint is idempotent.
    make_trainer(tmp_path, resume=True).train(until_unique_tokens=8)
    rows = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert len([row for row in rows if row["event"] == "snapshot"]) == 1


def test_planned_snapshots_survive_rolling_retention_and_are_portable(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train()
    assert len(candidate_checkpoint_paths(tmp_path / "run")) == 1
    paths = sorted((tmp_path / "run/snapshots").glob("*.safetensors"))
    assert [path.name for path in paths] == ["model_000000000008.safetensors", "model_000000000016.safetensors"]
    portable = tmp_path / "portable.safetensors"
    shutil.copyfile(paths[0], portable)
    result = load_model_weights(portable, model=make_model(), expected_experiment_config=trainer.config.to_dict())
    assert result["source_train_state"]["unique_tokens_seen"] == 8


def test_committed_snapshot_is_not_reused_for_different_weights(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.train(until_unique_tokens=8)
    path = tmp_path / "run/snapshots/model_000000000008.safetensors"
    metadata = snapshots.snapshot_metadata(path)
    with torch.no_grad():
        next(trainer.model.parameters()).add_(1)
    with pytest.raises(ValueError, match="weights differ"):
        snapshots.publish_snapshot(path, model=trainer.model, metadata=metadata)


def test_unknown_snapshot_metadata_fails_closed(tmp_path):
    trainer = make_trainer(tmp_path)
    path = tmp_path / "future.safetensors"
    snapshots.save_file(trainer.model.state_dict(), str(path), metadata={
        snapshots.METADATA_KEY: json.dumps({"snapshot_format_version": 99}),
    })
    original = path.read_bytes()
    with pytest.raises(ValueError, match="unsupported snapshot"):
        snapshots.publish_snapshot(path, model=trainer.model, metadata={})
    assert path.read_bytes() == original


def test_final_validation_is_not_duplicated_and_changed_settings_are_not_reused(tmp_path):
    trainer = make_trainer(tmp_path, eval_batches=1, eval_every_tokens=8)
    trainer.train(until_unique_tokens=8)
    trainer = make_trainer(tmp_path, resume=True, eval_batches=1, eval_every_tokens=8)
    trainer.train(until_unique_tokens=8)
    rows = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    assert len([row for row in rows if row["event"] == "validation"]) == 1
    trainer = make_trainer(tmp_path, resume=True, eval_batches=2, eval_every_tokens=8)
    trainer.train(until_unique_tokens=8)
    rows = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    assert [row["validation_blocks"] for row in rows if row["event"] == "validation"] == [1, 2]
