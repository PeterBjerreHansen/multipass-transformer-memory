import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as save_safetensors

from tiny_mistral_mptt.data.packed_dataset import StatefulBlockSampler
from tiny_mistral_mptt.training.checkpoint import (
    FORMAT_VERSION,
    TrainState,
    load_checkpoint,
    load_checkpoint_for_evaluation,
    load_model_weights,
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
        training_elapsed_seconds=12.5,
        phase="B",
    )
    path = save_checkpoint(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=state,
        experiment_config={"variant": "memory_attention", "memory_write_mode": "memory_token"},
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
    assert loaded_state.training_elapsed_seconds == 12.5
    assert restored_sampler.next_indices(8) == expected_next
    for name, tensor in replacement.state_dict().items():
        torch.testing.assert_close(tensor, expected_parameters[name], atol=0, rtol=0)


def test_checkpoint_refuses_resume_with_regenerated_data_manifest(tmp_path):
    model, optimizer = _objects()
    path = save_checkpoint(
        tmp_path / "old-data.pt",
        model=model,
        optimizer=optimizer,
        sampler_state={"position": 0},
        train_state=TrainState(),
        experiment_config={"variant": "vanilla"},
        data_manifest_sha256="old-padded-data",
    )
    replacement, replacement_optimizer = _objects()

    with pytest.raises(ValueError, match="data manifest changed across resume"):
        load_checkpoint(
            path,
            model=replacement,
            optimizer=replacement_optimizer,
            expected_manifest_sha256="new-unpadded-data",
        )


def test_current_checkpoint_without_elapsed_counter_resumes_at_zero(tmp_path):
    model, optimizer = _objects()
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "pre-timing-counter.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(unique_tokens_seen=8),
        experiment_config={"variant": "vanilla"},
        data_manifest_sha256="same",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["train_state"].pop("training_elapsed_seconds")
    torch.save(payload, path)

    replacement, replacement_optimizer = _objects()
    state, _ = load_checkpoint(
        path,
        model=replacement,
        optimizer=replacement_optimizer,
        expected_manifest_sha256="same",
    )
    assert state.training_elapsed_seconds == 0.0


def test_init_from_accepts_weights_only_snapshot_with_run_metadata(tmp_path):
    source, _ = _objects()
    snapshot_dir = tmp_path / "source-run" / "snapshots"
    snapshot_dir.mkdir(parents=True)
    snapshot = snapshot_dir / "model_000000000128.safetensors"
    save_safetensors(
        {
            name: tensor.detach().contiguous()
            for name, tensor in source.state_dict().items()
        },
        str(snapshot),
    )
    snapshot.with_suffix(".json").write_text(
        json.dumps(
            {
                "optimizer_steps": 16,
                "unique_tokens_seen": 128,
                "model_positions_seen": 128,
                "phase": "B",
                "variant": "vanilla",
            }
        ),
        encoding="utf-8",
    )
    (snapshot_dir.parent / "run.json").write_text(
        json.dumps({"config": {"variant": "vanilla"}}),
        encoding="utf-8",
    )

    replacement, _ = _objects()
    provenance = load_model_weights(
        snapshot,
        model=replacement,
        expected_experiment_config={"variant": "vanilla"},
    )

    assert provenance["source_format"] == "safetensors_snapshot"
    assert provenance["source_train_state"]["unique_tokens_seen"] == 128
    for name, tensor in replacement.state_dict().items():
        torch.testing.assert_close(
            tensor, source.state_dict()[name], atol=0, rtol=0
        )


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
            "variant": "memory_attention",
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
                "variant": "memory_attention",
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


def test_evaluation_checkpoint_accepts_public_variant_alias(tmp_path):
    model, optimizer = _objects()
    sampler = StatefulBlockSampler(5, seed=3)
    path = save_checkpoint(
        tmp_path / "memory-attention.pt",
        model=model,
        optimizer=optimizer,
        sampler_state=sampler.state_dict(),
        train_state=TrainState(),
        experiment_config={"variant": "memory_attention", "memory_write_mode": "dense"},
        data_manifest_sha256="same",
    )
    replacement, _ = _objects()
    load_checkpoint_for_evaluation(
        path,
        model=replacement,
        expected_manifest_sha256="same",
        expected_experiment_config={
            "variant": "memory_attention",
            "memory_write_mode": "dense",
        },
    )


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


@pytest.mark.parametrize("field,value", [
    ("memory_pattern", "strided"),
    ("memory_layers", [1]),
    ("recurrent_merger", "recirculation"),
    ("recurrent_layers", [1]),
])
def test_checkpoint_semantics_do_not_collapse_with_shared_attention_dispatch(tmp_path, field, value):
    model, optimizer = _objects()
    config = {
        "variant": "memory_attention", "memory_pattern": "dense",
        "memory_layers": [0], "recurrent_merger": "projected_residual",
        "recurrent_layers": [0],
    }
    path = save_checkpoint(
        tmp_path / "hybrid.pt", model=model, optimizer=optimizer,
        sampler_state=StatefulBlockSampler(5, seed=3).state_dict(),
        train_state=TrainState(), data_manifest_sha256="same", experiment_config=config,
    )
    changed = {**config, field: value}
    # Deliberately identical tensor structure: semantic checks must catch it first.
    with pytest.raises(ValueError, match=field):
        load_model_weights(path, model=model, expected_experiment_config=changed)
    with pytest.raises(ValueError, match=field):
        load_checkpoint(path, model=model, optimizer=optimizer,
                        expected_manifest_sha256="same", expected_experiment_config=changed)


@pytest.mark.parametrize("alias,explicit", [
    ("dense_memory_attention", {"memory_pattern": "dense"}),
    ("strided_memory_attention", {"memory_pattern": "strided", "memory_write_stride": 2}),
    ("dense_and_strided_memory_attention", {
        "memory_pattern": "dense_and_strided", "memory_dense_window": 2,
        "memory_sparse_window": 2, "memory_sparse_stride": 2,
    }),
])
def test_attention_preset_and_explicit_config_resume_equally(tmp_path, alias, explicit):
    model, optimizer = _objects()
    fields = {key: value for key, value in explicit.items() if key != "memory_pattern"}
    path = save_checkpoint(
        tmp_path / "attention.pt", model=model, optimizer=optimizer,
        sampler_state=StatefulBlockSampler(5, seed=3).state_dict(),
        train_state=TrainState(), data_manifest_sha256="same",
        experiment_config={"variant": alias, **fields},
    )
    target = {"variant": "memory_attention", **explicit}
    load_model_weights(path, model=model, expected_experiment_config=target)
    load_checkpoint(path, model=model, optimizer=optimizer,
                    expected_manifest_sha256="same", expected_experiment_config=target)
