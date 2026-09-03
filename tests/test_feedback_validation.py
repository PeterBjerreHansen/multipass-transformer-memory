import json

import pytest
import torch

from test_snapshot_recovery import make_trainer
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.training import feedback_validation
from tiny_mistral_mptt.training.checkpoint import candidate_checkpoint_paths, inspect_checkpoint
from tiny_mistral_mptt.training.journal import repair_metrics_to_checkpoint, append_jsonl
from tiny_mistral_mptt.training.snapshots import snapshot_metadata


def events(trainer, event):
    return [row for line in trainer.metrics_path.read_text().splitlines()
            if (row := json.loads(line))["event"] == event]


@pytest.mark.parametrize("settings,match", [
    ({"feedback_eval_at_tokens": [9], "snapshot_at_tokens": [8]}, "select thresholds"),
    ({"feedback_eval_max_blocks": 0}, "positive"),
    ({"feedback_eval_autocast_dtype": "float16"}, "config, float32 or bfloat16"),
])
def test_schedule_validation(settings, match):
    with pytest.raises(ValueError, match=match):
        ExperimentConfig.from_dict(settings).validate()


def test_selected_snapshots_only_and_training_unchanged(tmp_path, monkeypatch):
    original = feedback_validation.evaluate_feedback_nll
    calls = []
    trainer = make_trainer(tmp_path / "scheduled", feedback_eval_at_tokens=[7], eval_batches=1,
                           eval_every_tokens=8, eval_passes=4)
    def capture(*args, **kwargs):
        calls.append(trainer.state.unique_tokens_seen)
        snapshot = trainer.run_dir / "snapshots/model_000000000008.safetensors"
        assert snapshot_metadata(snapshot)["requested_thresholds"] == [7, 8]
        checkpoint = candidate_checkpoint_paths(trainer.run_dir)[0]
        assert inspect_checkpoint(checkpoint)["checkpoint_metadata"]["pending_snapshots"] == [7, 8]
        return original(*args, **kwargs)
    monkeypatch.setattr(feedback_validation, "evaluate_feedback_nll", capture)
    trainer.train()
    control = make_trainer(tmp_path / "control", eval_batches=1, eval_every_tokens=8, eval_passes=4)
    control.train()
    assert calls == [8]
    assert [row["unique_tokens_seen"] for row in events(trainer, "validation")] == [8, 16, 24]
    assert all(row["eval_passes"] == 4 for row in events(trainer, "validation"))
    rows = events(trainer, "feedback_validation")
    assert len(rows) == 1 and rows[0]["requested_thresholds"] == [7]
    assert rows[0]["result"]["predicted_tokens"] == 8
    assert rows[0]["result"]["aligned_predicted_tokens"] == 7
    assert len(list((trainer.run_dir / "snapshots").glob("*.safetensors"))) == 2
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, control.model.state_dict()[key], atol=0, rtol=0)
    assert trainer.sampler.state_dict() == control.sampler.state_dict()
    assert trainer.pass_scheduler.state_dict() == control.pass_scheduler.state_dict()
    for parameter, state in trainer.optimizer.state_dict()["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, control.optimizer.state_dict()["state"][parameter][key], atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["decode", "publish", "journal"])
def test_interrupted_feedback_recovers_before_training_and_reuses_completed_report(tmp_path, monkeypatch, failure):
    from tiny_mistral_mptt.training import trainer as trainer_module
    trainer = make_trainer(tmp_path, feedback_eval_at_tokens=[8])
    module = trainer_module if failure == "journal" else feedback_validation
    name = {"decode": "evaluate_feedback_nll", "publish": "atomic_write_json", "journal": "append_jsonl"}[failure]
    original = getattr(module, name)
    def fail(*args, **kwargs):
        if failure == "journal" and args[1].get("event") != "feedback_validation":
            return original(*args, **kwargs)
        if failure == "decode":
            raise InterruptedError("signal")
        raise OSError("interrupted feedback publication")
    monkeypatch.setattr(module, name, fail)
    if failure == "decode":
        trainer.train()
    else:
        with pytest.raises(OSError, match="interrupted feedback"):
            trainer.train()
    assert trainer.state.unique_tokens_seen == 8
    assert events(trainer, "feedback_validation") == []
    monkeypatch.setattr(module, name, original)
    calls = []
    original_eval = feedback_validation.evaluate_feedback_nll
    def capture(*args, **kwargs):
        calls.append(1)
        return original_eval(*args, **kwargs)
    monkeypatch.setattr(feedback_validation, "evaluate_feedback_nll", capture)
    recovered = make_trainer(tmp_path, resume=True, feedback_eval_at_tokens=[8])
    recovered.train(until_unique_tokens=8)
    assert len(calls) == (0 if failure == "journal" else 1)
    assert len(events(recovered, "feedback_validation")) == 1
    make_trainer(tmp_path, resume=True, feedback_eval_at_tokens=[8]).train(until_unique_tokens=8)
    assert len(calls) == (0 if failure == "journal" else 1)
    assert len(events(recovered, "feedback_validation")) == 1


def test_changed_subset_creates_separate_report_and_rollback_repairs_events(tmp_path):
    trainer = make_trainer(tmp_path, feedback_eval_at_tokens=[8])
    trainer.train(until_unique_tokens=8)
    trainer = make_trainer(tmp_path, resume=True, feedback_eval_at_tokens=[8], feedback_eval_max_blocks=2)
    trainer.train(until_unique_tokens=8)
    rows = events(trainer, "feedback_validation")
    assert [row["result"]["blocks"] for row in rows] == [1, 2]
    assert rows[0]["report_id"] != rows[1]["report_id"]
    append_jsonl(trainer.metrics_path, {"event": "feedback_validation", "unique_tokens_seen": 16})
    repair_metrics_to_checkpoint(trainer.metrics_path, trainer.state)
    assert len(events(trainer, "feedback_validation")) == 2


def test_signal_during_real_decode_leaves_durable_pending_work(tmp_path):
    trainer = make_trainer(tmp_path, feedback_eval_at_tokens=[8])
    polls = 0
    def stop():
        nonlocal polls
        polls += 1
        return polls >= 5
    trainer.stop_requested = stop
    trainer.train()
    assert trainer.state.unique_tokens_seen == 8
    assert trainer.model.training
    assert events(trainer, "feedback_validation") == []
    assert not list((trainer.run_dir / "evaluations").glob("*.json"))
    checkpoint = candidate_checkpoint_paths(trainer.run_dir)[0]
    assert inspect_checkpoint(checkpoint)["checkpoint_metadata"]["pending_snapshots"] == [7, 8]
    make_trainer(tmp_path, resume=True, feedback_eval_at_tokens=[8]).train(until_unique_tokens=8)
    assert len(events(trainer, "feedback_validation")) == 1
