from pathlib import Path
import json

import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.data.prepare import PreparationRequest, materialize_from_document_iterators
from tiny_mistral_mptt.data.recipes import DOLMINO_50B_SOURCES
from tiny_mistral_mptt.training.trainer import Trainer
from tiny_mistral_mptt.training.checkpoint import candidate_checkpoint_paths
from tiny_mistral_mptt.variants.vanilla import VanillaVariant


def fake_docs(offset: int):
    for doc in range(10_000):
        yield f"{offset}-{doc}-abcdefghijklmnopqrstuvwxyz"


def make_artifact(root: Path):
    materialize_from_document_iterators(
        PreparationRequest(
            output_dir=root,
            sequence_length=8,
            train_tokens=8 * 60,
            validation_tokens=8 * 12,
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


def make_model():
    torch.manual_seed(17)
    return VanillaVariant(MistralForCausalLM(micro_config(), attention_backend="reference"))


def test_end_to_end_vanilla_trainer_and_resume(tmp_path):
    data_dir = tmp_path / "data"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    first_cfg = ExperimentConfig(
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "run1"),
        device="cpu",
        dtype="float32",
        attention_backend="reference",
        batch_size=2,
        grad_accum_steps=1,
        max_unique_tokens=64,
        learning_rate=1e-4,
        eval_every_tokens=16,
        eval_batches=2,
        checkpoint_every_tokens=32,
    )
    first = Trainer(model=make_model(), config=first_cfg, train_data=train, validation_data=val, device=torch.device("cpu"))
    state1 = first.train(until_unique_tokens=32)
    assert state1.unique_tokens_seen == 32
    assert state1.training_elapsed_seconds > 0
    checkpoint = candidate_checkpoint_paths(tmp_path / "run1")[0]
    assert checkpoint.exists()
    run_info = json.loads((tmp_path / "run1" / "run.json").read_text())
    assert run_info["batching"] == {
        "linguistic_sequence_length": 8,
        "physical_sequence_length": 8,
        "microbatch_size": 2,
        "grad_accum_steps": 1,
        "microbatch_tokens": 16,
        "microbatch_model_positions": 16,
        "control_positions_per_microbatch": 0,
        "nominal_optimizer_batch_tokens": 16,
        "nominal_optimizer_batch_model_positions": 16,
        "planned_optimizer_steps": 4,
    }
    source = run_info["source"]
    if source["git_commit"] is None:
        # Source archives intentionally have no .git metadata.
        assert source["git_dirty"] is None
    else:
        assert len(source["git_commit"]) == 40
        assert isinstance(source["git_dirty"], bool)

    second_cfg = ExperimentConfig.from_dict({**first_cfg.to_dict(),
        "output_dir": str(tmp_path / "run2"),
        "resume_from": str(checkpoint),
    })
    second = Trainer(model=make_model(), config=second_cfg, train_data=train, validation_data=val, device=torch.device("cpu"))
    state2 = second.train()
    assert state2.unique_tokens_seen == 64
    assert state2.optimizer_steps > state1.optimizer_steps
    assert state2.token_equivalent_compute == state2.unique_tokens_seen
    assert state2.training_elapsed_seconds > state1.training_elapsed_seconds


def test_trainer_can_finish_with_partial_gradient_accumulation(tmp_path):
    data_dir = tmp_path / "data-partial"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")
    cfg = ExperimentConfig(
        model_dir="unused",
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "partial-run"),
        device="cpu",
        dtype="float32",
        attention_backend="reference",
        batch_size=1,
        grad_accum_steps=3,
        max_unique_tokens=8 * 5,
        learning_rate=1e-4,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )
    trainer = Trainer(model=make_model(), config=cfg, train_data=train, validation_data=val, device=torch.device("cpu"))
    state = trainer.train()
    assert state.unique_tokens_seen == 40
    assert state.micro_steps == 5
    assert state.optimizer_steps == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "partial-run" / "metrics.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "train"
    ]
    assert [record["optimizer_batch_tokens"] for record in records] == [24, 16]
    assert [record["nominal_optimizer_batch_tokens"] for record in records] == [24, 24]
    assert [record["accumulation_steps"] for record in records] == [3, 2]


def test_interrupted_resume_matches_uninterrupted_parameters(tmp_path):
    data_dir = tmp_path / "data-exact"
    make_artifact(data_dir)
    train = PackedTokenDataset(data_dir, "train")
    val = PackedTokenDataset(data_dir, "validation")

    common = dict(
        model_dir="unused",
        data_dir=str(data_dir),
        device="cpu",
        dtype="float32",
        attention_backend="reference",
        seed=99,
        batch_size=2,
        grad_accum_steps=1,
        max_unique_tokens=64,
        learning_rate=1e-4,
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=0,
    )

    full_model = make_model()
    full_cfg = ExperimentConfig(output_dir=str(tmp_path / "full"), **common)
    Trainer(model=full_model, config=full_cfg, train_data=train, validation_data=val, device=torch.device("cpu")).train()

    interrupted_model = make_model()
    interrupted_cfg = ExperimentConfig(output_dir=str(tmp_path / "interrupted"), **common)
    Trainer(
        model=interrupted_model,
        config=interrupted_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train(until_unique_tokens=32)
    checkpoint = candidate_checkpoint_paths(tmp_path / "interrupted")[0]

    resumed_model = make_model()
    resumed_cfg = ExperimentConfig.from_dict({
        **interrupted_cfg.to_dict(),
        "output_dir": str(tmp_path / "resumed"),
        "resume_from": str(checkpoint),
    })
    Trainer(
        model=resumed_model,
        config=resumed_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    ).train()

    assert full_model.state_dict().keys() == resumed_model.state_dict().keys()
    for name, tensor in full_model.state_dict().items():
        torch.testing.assert_close(tensor, resumed_model.state_dict()[name], atol=0, rtol=0)
