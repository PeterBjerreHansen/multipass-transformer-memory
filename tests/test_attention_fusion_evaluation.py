import pytest
import torch

from conftest import micro_config
from test_evaluation_contract import Rows
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.evaluation.attention_fusion import evaluate_attention_fusion
from tiny_mistral_mptt.variants.memory_attention import MemoryAttentionVariant


def make_model(fusion: str) -> MemoryAttentionVariant:
    torch.manual_seed(31)
    return MemoryAttentionVariant(
        MistralForCausalLM(micro_config(), attention_backend="reference"),
        memory_window=3,
        memory_write_mode="dense",
        memory_layers=[1],
        memory_position_encoding="none",
        memory_reader_initialization="aligned_gqa",
        memory_attention_fusion=fusion,
        memory_attention_controller_hidden_size=(
            None if fusion == "residual" else 4
        ),
        initialization_seed=12,
    )


@pytest.mark.parametrize(
    ("fusion", "has_alpha", "has_beta"),
    [
        ("residual", False, False),
        ("destination_gated", False, True),
        ("attention_gated", True, False),
        ("dual_gated", True, True),
    ],
)
def test_fusion_diagnostic_reports_each_feedback_pass_and_gate(fusion, has_alpha, has_beta):
    model = make_model(fusion)
    result = evaluate_attention_fusion(
        model, Rows(), device="cpu", passes=3, max_blocks=1
    )
    assert model.training
    assert result["fusion"] == fusion
    assert result["blocks"] == 1
    assert [(row["pass"], row["layer"]) for row in result["sites"]] == [
        (2, 1),
        (3, 1),
    ]
    for row in result["sites"]:
        assert row["valid_memory_tokens"] == 7
        assert row["attention_rms"] > 0
        assert row["destination_rms"] > 0
        if has_alpha:
            assert row["alpha"] is not None
            assert row["alpha"]["mean"] == pytest.approx(0.1)
        else:
            assert row["alpha"] is None
            assert row["attention_contribution_rms"] == row["attention_rms"]
        if has_beta:
            assert row["beta"] is not None
            assert row["beta"]["mean"] == pytest.approx(0.9)
        else:
            assert row["beta"] is None
            assert row["destination_contribution_rms"] == row["destination_rms"]


def test_fusion_diagnostic_rejects_non_attention_models():
    model = MistralForCausalLM(micro_config(), attention_backend="reference")
    with pytest.raises(ValueError, match="no Memory Attention fusion sites"):
        evaluate_attention_fusion(model, Rows(), device="cpu", passes=2)
