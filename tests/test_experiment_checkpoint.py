from pathlib import Path

import pytest
import torch

from tiny_mistral_mptt.data.packed_dataset import StatefulBlockSampler
from tiny_mistral_mptt.training.checkpoint import (
    FORMAT_VERSION,
    TrainState,
    load_checkpoint,
    load_checkpoint_for_evaluation,
    save_checkpoint,
)


def _objects():
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    return model, optimizer


def test_checkpoint_restores_model_optimizer_sampler_and_all_counters(tmp_path):
    torch.manual_seed(11)
    model, optimizer = _objects()
    x = torch.randn(2, 4)
    model(x).sum().backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    sampler = StatefulBlockSampler(13, seed=7)
    sampler.next_indices(6)
    state = TrainState(
        optimizer_steps=1,
        micro_steps=2,
        unique_tokens_seen=64,
        model_positions_seen=72,
        token_equivalent_compute=144,
        phase="B",
    )
    path = save_checkpoint(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=state,
        experiment_config={"variant": "bank", "memory_write_mode": "memory_token"},
        data_manifest_sha256="manifest-hash",
    )
    expected_parameters = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    expected_next = sampler.next_indices(8)

    replacement, replacement_optimizer = _objects()
    loaded_state, sampler_state = load_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="manifest-hash",
    )
    restored_sampler = StatefulBlockSampler(13, seed=0)
    restored_sampler.load_state_dict(sampler_state)
    assert loaded_state == state
    assert restored_sampler.next_indices(8) == expected_next
    for name, tensor in replacement.state_dict().items():
        torch.testing.assert_close(tensor, expected_parameters[name], atol=0, rtol=0)


def test_checkpoint_rejects_training_config_changes(tmp_path):
    model, optimizer = _objects()
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(),
        experiment_config={"variant": "vanilla", "batch_size": 1, "output_dir": "a"},
        data_manifest_sha256="same",
    )
    replacement, replacement_optimizer = _objects()
    with pytest.raises(ValueError, match="batch_size"):
        load_checkpoint(
            path,
            model=replacement,
            optimizer=replacement_optimizer,
            expected_manifest_sha256="same",
            expected_experiment_config={"variant": "vanilla", "batch_size": 2, "output_dir": "b"},
        )


def test_evaluation_checkpoint_rejects_semantic_config_changes(tmp_path):
    model, optimizer = _objects()
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(),
        experiment_config={
            "variant": "bank",
            "memory_write_mode": "dense",
            "memory_layers": [3, 7],
        },
        data_manifest_sha256="same",
    )
    replacement, _ = _objects()
    with pytest.raises(ValueError, match="experiment config changed"):
        load_checkpoint_for_evaluation(
            path,
            model=replacement,
            expected_manifest_sha256="same",
            expected_experiment_config={
                "variant": "bank",
                "memory_write_mode": "periodic",
                "memory_layers": [3, 7],
            },
        )


def test_evaluation_checkpoint_loads_strictly_after_manifest_and_config_checks(tmp_path):
    model, optimizer = _objects()
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(unique_tokens_seen=8),
        experiment_config={"variant": "vanilla", "phase": "B"},
        data_manifest_sha256="same",
    )
    replacement, _ = _objects()
    metadata = load_checkpoint_for_evaluation(
        path,
        model=replacement,
        expected_manifest_sha256="same",
        expected_experiment_config={"variant": "vanilla", "phase": "B"},
    )
    assert metadata["path"] == str(path)
    assert metadata["train_state"]["unique_tokens_seen"] == 8
    for name, tensor in replacement.state_dict().items():
        torch.testing.assert_close(tensor, model.state_dict()[name], atol=0, rtol=0)


def test_constant_lr_resume_may_extend_stopping_budget(tmp_path):
    model, optimizer = _objects()
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "extend.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(unique_tokens_seen=4, model_positions_seen=4),
        experiment_config={
            "variant": "vanilla",
            "max_unique_tokens": 4,
            "lr_schedule": {"type": "constant"},
        },
        data_manifest_sha256="same",
    )
    replacement, replacement_optimizer = _objects()
    state, _ = load_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="same",
        expected_experiment_config={
            "variant": "vanilla",
            "max_unique_tokens": 8,
            "lr_schedule": {"type": "constant"},
        },
    )
    assert state.unique_tokens_seen == 4


def test_scheduled_resume_rejects_changed_horizon(tmp_path):
    model, optimizer = _objects()
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "scheduled.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(unique_tokens_seen=4, model_positions_seen=4),
        experiment_config={
            "variant": "vanilla",
            "max_unique_tokens": 4,
            "lr_schedule": {"type": "cosine"},
        },
        data_manifest_sha256="same",
    )
    replacement, replacement_optimizer = _objects()
    with pytest.raises(ValueError, match="max_unique_tokens"):
        load_checkpoint(
            path,
            model=replacement,
            optimizer=replacement_optimizer,
            expected_manifest_sha256="same",
            expected_experiment_config={
                "variant": "vanilla",
                "max_unique_tokens": 8,
                "lr_schedule": {"type": "cosine"},
            },
        )


def test_output_and_operational_checkpoint_schedule_are_relocatable(tmp_path):
    model, optimizer = _objects()
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "ops.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(unique_tokens_seen=7, model_positions_seen=7),
        experiment_config={
            "variant": "memory_add",
            "phase": "B",
            "output_dir": "old",
            "checkpoint_every_tokens": 64,
            "checkpoint_every_seconds": 30,
            "checkpoint_keep_last": 2,
            "snapshot_at_tokens": [100],
        },
        data_manifest_sha256="same",
    )
    replacement, replacement_optimizer = _objects()
    state, _ = load_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="same",
        expected_experiment_config={
            "variant": "memory_add",
            "phase": "B",
            "output_dir": "new",
            "checkpoint_every_tokens": 1000,
            "checkpoint_every_seconds": 600,
            "checkpoint_keep_last": 4,
            "snapshot_at_tokens": [200],
        },
    )
    assert state.unique_tokens_seen == 7


def test_clean_break_rejects_old_checkpoint_format(tmp_path):
    model, optimizer = _objects()
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "current.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(),
        experiment_config={"variant": "vanilla"},
        data_manifest_sha256="same",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == FORMAT_VERSION
    payload["format_version"] = FORMAT_VERSION - 1
    old = tmp_path / "old.pt"
    torch.save(payload, old)
    replacement, replacement_optimizer = _objects()
    with pytest.raises(ValueError, match="unsupported experiment checkpoint format"):
        load_checkpoint(
            old,
            model=replacement,
            optimizer=replacement_optimizer,
            expected_manifest_sha256="same",
        )
