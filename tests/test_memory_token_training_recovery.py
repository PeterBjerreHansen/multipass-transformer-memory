from __future__ import annotations

from pathlib import Path

import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.data.prepare import PreparationRequest, materialize_from_document_iterators
from tiny_mistral_mptt.data.recipes import DOLMINO_50B_SOURCES
from tiny_mistral_mptt.training.trainer import Trainer
from tiny_mistral_mptt.variants.memory_attention import MemoryAttentionVariant
from tiny_mistral_mptt.variants.memory_attention_recurrent_hybrid import MemoryAttentionRecurrentHybridVariant


def _docs(offset: int):
    for doc in range(10_000):
        yield f"{offset}-{doc}-abcdefghijklmnopqrstuvwxyz"


def _artifact(root: Path):
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
        ),
        iterators={source.name: iter(_docs(i)) for i, source in enumerate(DOLMINO_50B_SOURCES)},
        tokenize=lambda text: [3 + (ord(ch) % 80) for ch in text],
    )


def _model(hybrid: bool):
    torch.manual_seed(444)
    backbone = MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=4),
        attention_backend="reference",
    )
    cls = MemoryAttentionRecurrentHybridVariant if hybrid else MemoryAttentionVariant
    return cls(
        backbone,
        **({"recurrent_merger": "projected_residual", "recurrent_layers": [0]} if hybrid else {}),
        memory_window=3,
        memory_write_mode="memory_token",
        memory_write_stride=2,
        memory_token_visibility="write_only",
        initialization_seed=909,
    )


def _assert_optimizer_equal(a: torch.optim.Optimizer, b: torch.optim.Optimizer) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["param_groups"] == sb["param_groups"]
    assert sa["state"].keys() == sb["state"].keys()
    for key in sa["state"]:
        assert sa["state"][key].keys() == sb["state"][key].keys()
        for field, value in sa["state"][key].items():
            other = sb["state"][key][field]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, atol=0, rtol=0)
            else:
                assert value == other


@pytest.mark.parametrize("hybrid", [False, True])
def test_memory_token_training_is_bit_exact_across_auto_resume(tmp_path, hybrid):
    data_dir = tmp_path / "data"
    _artifact(data_dir)
    train = load_packed_dataset_for_experiment(
        data_dir, "train", memory_write_mode="memory_token", memory_write_stride=2
    )
    val = load_packed_dataset_for_experiment(
        data_dir, "validation", memory_write_mode="memory_token", memory_write_stride=2
    )
    assert train.sequence_length == 11
    assert train.linguistic_sequence_length == 8

    common = dict(
        variant="memory_attention",
        **({"recurrent_merger": "projected_residual", "recurrent_layers": [0]} if hybrid else {}),
        phase="B",
        model_dir="unused",
        data_dir=str(data_dir),
        device="cpu",
        attention_backend="reference",
        seed=1337,
        architecture_seed=909,
        batch_size=1,
        grad_accum_steps=1,
        max_unique_tokens=64,
        learning_rate=1e-4,
        pretrained_learning_rate=1e-4,
        added_learning_rate=1e-4,
        lr_schedule={"type": "constant"},
        pass_schedule=[{"probabilities": {2: 1.0}}],
        pass_loss_weights=[0.25, 0.75],
        memory_window=3,
        memory_write_mode="memory_token",
        memory_write_stride=2,
        memory_token_visibility="write_only",
        eval_every_tokens=0,
        eval_batches=0,
        checkpoint_every_tokens=16,
        checkpoint_every_seconds=0.0,
        checkpoint_keep_last=2,
    )

    full_cfg = ExperimentConfig(output_dir=str(tmp_path / "full"), **common)
    full = Trainer(
        model=_model(hybrid),
        config=full_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
    )
    full_state = full.train()

    resumed_cfg = ExperimentConfig(output_dir=str(tmp_path / "resumed"), **common)
    first_segment = Trainer(
        model=_model(hybrid),
        config=resumed_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
        resume_auto=True,
    )
    interrupted_state = first_segment.train(until_unique_tokens=32)
    assert interrupted_state.unique_tokens_seen == 32
    assert interrupted_state.model_positions_seen == 44

    second_segment = Trainer(
        model=_model(hybrid),
        config=resumed_cfg,
        train_data=train,
        validation_data=val,
        device=torch.device("cpu"),
        resume_auto=True,
    )
    resumed_state = second_segment.train()

    assert resumed_state == full_state
    assert resumed_state.unique_tokens_seen == 64
    assert resumed_state.model_positions_seen == 88
    assert resumed_state.token_equivalent_compute == 176
    assert full.model.state_dict().keys() == second_segment.model.state_dict().keys()
    for name, tensor in full.model.state_dict().items():
        torch.testing.assert_close(
            tensor, second_segment.model.state_dict()[name], atol=0, rtol=0
        )
    _assert_optimizer_equal(full.optimizer, second_segment.optimizer)
