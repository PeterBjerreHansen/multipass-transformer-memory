from pathlib import Path

import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset
from tiny_mistral_mptt.data.prepare import PreparationRequest, materialize_from_document_iterators
from tiny_mistral_mptt.data.recipes import DOLMINO_50B_SOURCES
from tiny_mistral_mptt.evaluation.nmp import evaluate_nmp
from tiny_mistral_mptt.model_factory import build_variant


def _documents(offset: int):
    for index in range(1000):
        yield f"{offset}-{index}-abcdefghijklmnopqrstuvwxyz"


def _artifact(root: Path) -> None:
    materialize_from_document_iterators(
        PreparationRequest(
            output_dir=root,
            sequence_length=8,
            train_tokens=8 * 8,
            validation_tokens=8 * 8,
            seed=5,
            dataset_repo="fake",
            requested_revision="x",
            resolved_revision="x",
            tokenizer_file=Path("fake-tokenizer"),
            tokenizer_sha256="fake",
            vocab_size=97,
            bos_token_id=1,
        ),
        iterators={
            source.name: iter(_documents(index))
            for index, source in enumerate(DOLMINO_50B_SOURCES)
        },
        tokenize=lambda text: [3 + (ord(character) % 80) for character in text],
    )


def test_nmp_evaluator_is_deterministic_and_reports_fixed_k_metrics(tmp_path):
    data_dir = tmp_path / "data"
    _artifact(data_dir)
    dataset = PackedTokenDataset(data_dir, "validation")
    torch.manual_seed(44)
    model = build_variant(
        "bank",
        MistralForCausalLM(micro_config(), attention_backend="reference"),
        memory_write_mode="periodic",
        memory_write_stride=2,
        memory_layers=[0],
        bank_nmp_weight=0.1,
    )
    model.train()
    first = evaluate_nmp(
        model,
        dataset,
        device="cpu",
        passes=2,
        bank_nmp_loss_weights=[0.0, 1.0],
        max_blocks=2,
    )
    second = evaluate_nmp(
        model,
        dataset,
        device="cpu",
        passes=2,
        bank_nmp_loss_weights=[0.0, 1.0],
        max_blocks=2,
    )
    assert model.training
    assert first.passes == 2
    assert first.blocks == 2
    assert first.predicted_tokens == 14
    assert first.metrics == pytest.approx(second.metrics)
    assert "pass_2_loss" in first.metrics
    assert "bank_nmp_loss" in first.metrics
    assert "bank_nmp_event_target_rms" in first.metrics
    assert "bank_nmp_query_target_rms" in first.metrics
    assert first.metrics["bank_nmp_valid_events"] > 0
