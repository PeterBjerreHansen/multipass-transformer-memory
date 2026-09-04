import json
from pathlib import Path

import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.data.manifest import file_sha256
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.data.prepare import PreparationRequest, materialize_from_document_iterators
from tiny_mistral_mptt.data.recipes import DOLMINO_50B_SOURCES
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.training.trainer import Trainer
from tiny_mistral_mptt.training.checkpoint import candidate_checkpoint_paths
from tiny_mistral_mptt.variants.fbt import FBTVariant


def fake_docs(offset: int):
    for doc in range(10_000):
        yield f"{offset}-{doc}-abcdefghijklmnopqrstuvwxyz"


def make_artifact(root: Path):
    materialize_from_document_iterators(
        PreparationRequest(
            output_dir=root,
            sequence_length=8,
            train_tokens=8 * 40,
            validation_tokens=8 * 8,
            seed=5,
            dataset_repo="fake",
            requested_revision="x",
            resolved_revision="x",
            tokenizer_file=Path("fake-tokenizer"),
            tokenizer_sha256="fake",
            vocab_size=97,
            bos_token_id=1,
            forbidden_token_ids=(96,),
        ),
        iterators={source.name: iter(fake_docs(i)) for i, source in enumerate(DOLMINO_50B_SOURCES)},
        tokenize=lambda text: [3 + (ord(ch) % 80) for ch in text],
    )


def make_fbt(seed=123):
    torch.manual_seed(17)
    return FBTVariant(
        MistralForCausalLM(micro_config(), attention_backend="reference"),
        initialization_seed=seed,
    )


def make_adaptive_recirculation(seed=123):
    torch.manual_seed(17)
    return build_variant(
        "recirculation",
        MistralForCausalLM(micro_config(), attention_backend="reference"),
        architecture_seed=seed,
        recirculation_source_layer=1,
        recirculation_destination_layer=0,
        recirculation_mode="adaptive",
    )



def test_phase_a_fixed_two_pass_training_counts_compute_and_freezes_backbone(tmp_path):
    data_dir = tmp_path / "data"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    model = make_fbt()
    before = {name: tensor.detach().clone() for name, tensor in model.backbone.state_dict().items()}
    cfg = ExperimentConfig(
        variant="fbt",
        phase="A",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "phase-a"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=32,
        batch_size=1,
        grad_accum_steps=1,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pass_loss_weights=[0.0, 1.0],
        added_learning_rate=1e-3,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    state = Trainer(model=model, config=cfg, train_data=train, validation_data=val, device=torch.device("cpu")).train()
    assert state.unique_tokens_seen == 32
    assert state.token_equivalent_compute == 64
    for name, tensor in model.backbone.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def test_phase_b_has_independent_pretrained_and_added_learning_rates(tmp_path):
    data_dir = tmp_path / "data-lr"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    cfg = ExperimentConfig(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "phase-b"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pretrained_learning_rate=1e-6,
        added_learning_rate=1e-4,
        pretrained_weight_decay=0.01,
        added_weight_decay=1e-4,
        lr_schedule={"type": "constant"},
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    trainer = Trainer(model=make_fbt(), config=cfg, train_data=train, validation_data=val, device=torch.device("cpu"))
    groups = {group["group_name"]: group["base_lr"] for group in trainer.optimizer.param_groups}
    assert groups == {"pretrained": 1e-6, "added": 1e-4}
    decay = {
        group["group_name"]: group["weight_decay"]
        for group in trainer.optimizer.param_groups
    }
    assert decay == {"pretrained": 0.01, "added": 1e-4}


def test_integrated_freeze_unfreezes_without_resetting_added_optimizer_state(
    tmp_path,
):
    data_dir = tmp_path / "data-integrated-freeze"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    run_dir = tmp_path / "integrated-freeze"
    cfg = ExperimentConfig(
        variant="recirculation",
        recirculation_source_layer=1,
        recirculation_destination_layer=0,
        recirculation_mode="adaptive",
        training_forward="parallel_multipass",
        pass_schedule=[{"probabilities": {2: 1.0}}],
        freeze_pretrained_until_tokens=16,
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(run_dir),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=32,
        batch_size=1,
        grad_accum_steps=1,
        pretrained_learning_rate=1e-3,
        added_learning_rate=1e-3,
        lr_schedule={"type": "constant"},
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    first_model = make_adaptive_recirculation()
    initial_backbone = {
        name: value.detach().clone()
        for name, value in first_model.backbone.state_dict().items()
    }
    first = Trainer(
        model=first_model,
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    )
    first.train(until_unique_tokens=16)

    for name, value in first_model.backbone.state_dict().items():
        torch.testing.assert_close(value, initial_backbone[name], atol=0, rtol=0)
    checkpoint = candidate_checkpoint_paths(run_dir)[0]
    added_parameter = next(iter(first_model.added_parameters()))
    first_added_step = int(first.optimizer.state[added_parameter]["step"].item())
    assert first_added_step == 2
    pretrained_group = next(
        group for group in first.optimizer.param_groups if group["group_name"] == "pretrained"
    )
    assert all(parameter not in first.optimizer.state for parameter in pretrained_group["params"])

    resumed_cfg = ExperimentConfig.from_dict(
        {**cfg.to_dict(), "resume_from": str(checkpoint)}
    )
    resumed_model = make_adaptive_recirculation()
    resumed = Trainer(
        model=resumed_model,
        config=resumed_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
        allow_source_mismatch=True,
    )
    assert all(parameter.requires_grad for parameter in resumed_model.backbone.parameters())
    resumed_added = next(iter(resumed_model.added_parameters()))
    assert int(resumed.optimizer.state[resumed_added]["step"].item()) == first_added_step

    resumed.train()

    assert int(resumed.optimizer.state[resumed_added]["step"].item()) == 4
    assert any(
        not torch.equal(value, initial_backbone[name])
        for name, value in resumed_model.backbone.state_dict().items()
    )
    records = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    train_records = [record for record in records if record["event"] == "train"]
    assert [record["backbone_frozen"] for record in train_records] == [True, True, False, False]


def test_training_journal_can_aggregate_over_token_intervals(tmp_path):
    data_dir = tmp_path / "data-log-cadence"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    cfg = ExperimentConfig(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "run-log-cadence"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=64,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        train_log_every_tokens=16,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    Trainer(
        model=make_fbt(),
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()

    records = [
        json.loads(line)
        for line in (tmp_path / "run-log-cadence" / "metrics.jsonl").read_text().splitlines()
    ]
    train_records = [record for record in records if record["event"] == "train"]
    assert len(train_records) == 4
    assert [record["log_interval_tokens"] for record in train_records] == [16, 16, 16, 16]
    assert [record["log_interval_updates"] for record in train_records] == [2, 2, 2, 2]
    assert all(record["unique_tokens_seen"] == (index + 1) * 16 for index, record in enumerate(train_records))
    cumulative_times = [record["training_elapsed_seconds"] for record in train_records]
    assert cumulative_times == sorted(cumulative_times)
    assert all(value > 0 for value in cumulative_times)


def test_mixed_k_telemetry_uses_conditional_metric_counts_and_total_throughput(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "data-mixed-telemetry"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    model = make_fbt(seed=81)
    observed: dict[str, list[float]] = {}
    original_compute_loss = model.compute_loss

    def traced_compute_loss(input_ids, **kwargs):
        output = original_compute_loss(input_ids, **kwargs)
        for key, value in output.metrics.items():
            observed.setdefault(key, []).append(float(value))
        return output

    monkeypatch.setattr(model, "compute_loss", traced_compute_loss)
    cfg = ExperimentConfig(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "mixed-telemetry"),
        device="cpu",
        attention_backend="reference",
        seed=15,
        max_unique_tokens=64,
        pass_schedule=[{"probabilities": {2: 0.5, 3: 0.5}}],
        train_log_every_tokens=64,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    Trainer(
        model=model,
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()

    records = [
        json.loads(line)
        for line in (tmp_path / "mixed-telemetry" / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    record = next(item for item in records if item["event"] == "train")
    assert "pass_3_loss" in observed
    assert record["metric_observation_counts"]["pass_3_loss"] == len(
        observed["pass_3_loss"]
    )
    assert record["pass_3_loss"] == pytest.approx(
        sum(observed["pass_3_loss"]) / len(observed["pass_3_loss"])
    )
    assert record["tokens_per_second"] == pytest.approx(
        record["log_interval_tokens"] / record["log_interval_elapsed_seconds"]
    )
    assert sum(record["pass_histogram_interval"].values()) == 8


def test_signal_flushes_partial_training_log_window(tmp_path):
    data_dir = tmp_path / "data-partial-log"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    cfg = ExperimentConfig(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "partial-log"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=64,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        train_log_every_tokens=64,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    Trainer(
        model=make_fbt(),
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
        stop_requested=lambda: True,
    ).train()

    records = [
        json.loads(line)
        for line in (tmp_path / "partial-log" / "metrics.jsonl")
        .read_text()
        .splitlines()
    ]
    record = next(item for item in records if item["event"] == "train")
    assert record["log_interval_partial"] is True
    assert record["log_interval_tokens"] == 8
    assert record["log_interval_updates"] == 1


def test_init_from_loads_model_only_into_fresh_phase_b_run(tmp_path):
    data_dir = tmp_path / "data-init"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    phase_a_cfg = ExperimentConfig(
        variant="fbt",
        phase="A",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "a"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pass_loss_weights=[0.0, 1.0],
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    phase_a_model = make_fbt(seed=7)
    Trainer(
        model=phase_a_model,
        config=phase_a_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()
    checkpoint = candidate_checkpoint_paths(tmp_path / "a")[0]
    expected = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]

    phase_b_cfg = ExperimentConfig(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "b"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        init_from=str(checkpoint),
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    fresh = make_fbt(seed=999)
    trainer = Trainer(
        model=fresh,
        config=phase_b_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    )
    assert trainer.state.unique_tokens_seen == 0
    assert trainer.state.optimizer_steps == 0
    run_info = json.loads((tmp_path / "b" / "run.json").read_text(encoding="utf-8"))
    assert run_info["initialization_provenance"]["source_sha256"] == file_sha256(checkpoint)
    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0)


def test_mixed_pass_schedule_resume_is_bit_exact(tmp_path):
    data_dir = tmp_path / "data-resume"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    common = dict(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        device="cpu",
        attention_backend="reference",
        seed=55,
        architecture_seed=77,
        batch_size=1,
        grad_accum_steps=1,
        max_unique_tokens=64,
        pass_schedule=[{"probabilities": {1: 0.4, 2: 0.4, 3: 0.2}}],
        pass_loss_weights_by_k={
            1: [1.0],
            2: [0.25, 0.75],
            3: [0.05, 0.20, 0.75],
        },
        pretrained_learning_rate=1e-4,
        added_learning_rate=1e-4,
        lr_schedule={"type": "constant"},
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )

    full = make_fbt(seed=77)
    full_cfg = ExperimentConfig(output_dir=str(tmp_path / "full"), **common)
    full_state = Trainer(
        model=full,
        config=full_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()

    interrupted = make_fbt(seed=77)
    interrupted_cfg = ExperimentConfig(output_dir=str(tmp_path / "interrupted"), **common)
    Trainer(
        model=interrupted,
        config=interrupted_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train(until_unique_tokens=32)
    checkpoint = candidate_checkpoint_paths(tmp_path / "interrupted")[0]

    resumed = make_fbt(seed=77)
    resumed_cfg = ExperimentConfig.from_dict(
        {
            **interrupted_cfg.to_dict(),
            "output_dir": str(tmp_path / "resumed"),
            "resume_from": str(checkpoint),
        }
    )
    resumed_state = Trainer(
        model=resumed,
        config=resumed_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()

    assert resumed_state == full_state
    for name, tensor in full.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[name], atol=0, rtol=0)

    records = [
        json.loads(line)
        for line in (tmp_path / "full" / "metrics.jsonl").read_text().splitlines()
    ]
    train_records = [record for record in records if record["event"] == "train"]
    assert train_records
    final_histogram = train_records[-1]["pass_histogram"]
    assert sum(final_histogram.values()) == train_records[-1]["pass_samples"]
    assert set(final_histogram) <= {"1", "2", "3"}


def test_mixed_pass_schedule_forwards_k_specific_weights(tmp_path, monkeypatch):
    data_dir = tmp_path / "data-k-weights"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    model = make_fbt(seed=91)
    observed = []
    original_compute_loss = model.compute_loss

    def traced_compute_loss(input_ids, **kwargs):
        observed.append((kwargs["passes"], tuple(kwargs["loss_weights"])))
        return original_compute_loss(input_ids, **kwargs)

    monkeypatch.setattr(model, "compute_loss", traced_compute_loss)
    cfg = ExperimentConfig(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "run-k-weights"),
        device="cpu",
        attention_backend="reference",
        seed=15,
        max_unique_tokens=64,
        pass_schedule=[{"probabilities": {2: 0.5, 3: 0.5}}],
        pass_loss_weights_by_k={
            2: [0.25, 0.75],
            3: [0.05, 0.20, 0.75],
        },
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    Trainer(
        model=model,
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()

    assert observed
    expected = {
        2: (0.25, 0.75),
        3: (0.05, 0.20, 0.75),
    }
    assert all(weights == expected[passes] for passes, weights in observed)


def test_validation_gates_checkpoint_and_stop_at_first_passing_evaluation(tmp_path):
    data_dir = tmp_path / "data-early-stop"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    cfg = ExperimentConfig(
        variant="fbt",
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "early-stop"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=32,
        pass_schedule=[{"probabilities": {1: 1.0}}],
        ntp_pass_loss_weights_by_k={1: [1.0]},
        eval_every_tokens=8,
        eval_batches=1,
        eval_passes=4,
        early_stop={"pass_nll_max": {1: 100.0, 4: 100.0}},
        checkpoint_every_tokens=0,
    )
    trainer = Trainer(
        model=make_fbt(),
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    )

    state = trainer.train()

    assert state.unique_tokens_seen == 8
    records = [
        json.loads(line)
        for line in (tmp_path / "early-stop" / "metrics.jsonl").read_text().splitlines()
    ]
    validation = next(record for record in records if record["event"] == "validation")
    assert validation["early_stop"]["all_passed"] is True
    segments = [
        json.loads(line)
        for line in (tmp_path / "early-stop" / "segments.jsonl").read_text().splitlines()
    ]
    assert segments[-1]["reason"] == "validation_gates"
    checkpoint = candidate_checkpoint_paths(tmp_path / "early-stop")[0]

    resumed_cfg = ExperimentConfig.from_dict(
        {**cfg.to_dict(), "resume_from": str(checkpoint)}
    )
    resumed_state = Trainer(
        model=make_fbt(),
        config=resumed_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()
    assert resumed_state.unique_tokens_seen == 8
    resumed_records = [
        json.loads(line)
        for line in (tmp_path / "early-stop" / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(record["event"] == "train" for record in resumed_records) == 1


def test_memory_phase_a_runs_through_shared_trainer(tmp_path):
    from tiny_mistral_mptt.variants.memory_attention import MemoryAttentionVariant

    data_dir = tmp_path / "data-memory"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    torch.manual_seed(23)
    model = MemoryAttentionVariant(
        MistralForCausalLM(micro_config(), attention_backend="reference"),
        memory_window=4,
        memory_write_mode="dense",
        memory_write_stride=1,
        initialization_seed=31,
    )
    cfg = ExperimentConfig(
        variant="memory_attention",
        memory_write_mode="dense",
        phase="A",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "memory-a"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pass_loss_weights=[0.0, 1.0],
        added_learning_rate=1e-3,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    state = Trainer(
        model=model,
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()
    assert state.unique_tokens_seen == 16
    assert state.token_equivalent_compute == 32


def test_memory_add_phase_a_runs_through_shared_trainer(tmp_path):
    from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant

    data_dir = tmp_path / "data-memory-add"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    torch.manual_seed(29)
    model = MemoryAddVariant(
        MistralForCausalLM(micro_config(), attention_backend="reference")
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.backbone.state_dict().items()
    }
    cfg = ExperimentConfig(
        variant="memory_add",
        phase="A",
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "memory-add-a"),
        device="cpu",
        attention_backend="reference",
        max_unique_tokens=16,
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pass_loss_weights=[0.0, 1.0],
        added_learning_rate=1e-3,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    state = Trainer(
        model=model,
        config=cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()
    assert state.unique_tokens_seen == 16
    assert state.token_equivalent_compute == 32
    assert torch.count_nonzero(model.memory_projection.weight) > 0
    for name, tensor in model.backbone.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)
