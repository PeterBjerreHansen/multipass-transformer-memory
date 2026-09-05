import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.evaluation.feedback_continuation import (
    evaluate_feedback_continuation,
)
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.memory_attention import MemoryAttentionVariant


class TinyDataset:
    def __init__(self):
        self.rows = torch.tensor(
            [
                [1, 7, 3, 14, 22, 9, 31, 4, 51, 12],
                [2, 5, 8, 11, 17, 23, 29, 35, 41, 47],
            ]
        )
        self.sequence_length = int(self.rows.shape[1])

    def __len__(self):
        return int(self.rows.shape[0])

    def batch(self, indices, *, device):
        return self.rows[indices].to(device)


def make_model():
    torch.manual_seed(444)
    backbone = MistralForCausalLM(
        micro_config(sliding_window=4), attention_backend="reference"
    )
    model = MemoryAddVariant(backbone).eval()
    with torch.no_grad():
        model.memory_projection.weight.copy_(
            0.05 * torch.eye(model.config.hidden_size)
        )
    return model


def make_memory_attention_model():
    torch.manual_seed(445)
    backbone = MistralForCausalLM(
        micro_config(sliding_window=4), attention_backend="reference"
    )
    model = MemoryAttentionVariant(
        backbone,
        memory_window=3,
        memory_write_mode="strided",
        memory_write_stride=2,
        memory_layers=[1],
        initialization_seed=910,
    ).eval()
    with torch.no_grad():
        model.memory_readers["1"].o_proj.weight.copy_(
            torch.eye(model.config.hidden_size)
        )
    return model


def test_feedback_eval_k1_uses_feedback_after_the_shared_prompt_prediction():
    result = evaluate_feedback_continuation(
        make_model(),
        TinyDataset(),
        device="cpu",
        prefill_passes=1,
        prompt_tokens=4,
        continuation_tokens=6,
        horizons=[1, 2, 6],
    )
    assert result.prefill_passes == 1
    assert result.predicted_tokens_per_mode == 12
    assert result.exact_k_pass_nll_by_offset[0] == result.live_feedback_nll_by_offset[0]
    assert result.exact_to_live_kl_by_offset[0] == 0.0
    assert result.top1_agreement_by_offset[0] == 1.0
    assert any(
        exact != recurrent
        for exact, recurrent in zip(
            result.exact_k_pass_nll_by_offset[1:],
            result.live_feedback_nll_by_offset[1:],
            strict=True,
        )
    )
    torch.testing.assert_close(
        torch.tensor(result.exact_k_pass_nll_by_offset),
        torch.tensor(result.standard_k1_nll_by_offset),
        atol=0,
        rtol=0,
    )
    assert max(result.hidden_delta_rms_by_step) > 1e-5


def test_feedback_eval_k2_has_exact_initial_handoff_then_measures_drift():
    result = evaluate_feedback_continuation(
        make_model(),
        TinyDataset(),
        device="cpu",
        prefill_passes=2,
        prompt_tokens=4,
        continuation_tokens=6,
        horizons=[1, 2, 4, 6],
    )
    # Prefill predicts offset 0 identically; after consuming offset 0 the
    # recurrent and exact states are still identical, so offset 1 is identical
    # as well.  The feedback source differs when offset 1 itself is processed.
    assert abs(result.exact_k_pass_nll_by_offset[0] - result.live_feedback_nll_by_offset[0]) < 1e-7
    assert abs(result.exact_k_pass_nll_by_offset[1] - result.live_feedback_nll_by_offset[1]) < 1e-7
    assert result.hidden_delta_rms_by_step[0] < 1e-6
    assert max(result.hidden_delta_rms_by_step[1:]) > 1e-5
    assert [item.horizon for item in result.horizons] == [1, 2, 4, 6]
    two_step = result.horizons[1]
    expected_rms = (
        sum(value * value for value in result.hidden_delta_rms_by_step[:2]) / 2
    ) ** 0.5
    expected_cosine = sum(result.hidden_cosine_by_step[:2]) / 2
    assert two_step.hidden_delta_rms == pytest.approx(expected_rms)
    assert two_step.hidden_cosine == pytest.approx(expected_cosine)


def test_feedback_eval_exercises_strided_memory_attention_after_prefill():
    result = evaluate_feedback_continuation(
        make_memory_attention_model(),
        TinyDataset(),
        device="cpu",
        prefill_passes=3,
        prompt_tokens=4,
        continuation_tokens=6,
        horizons=[1, 2, 4, 6],
    )
    assert result.predicted_tokens_per_mode == 12
    assert all(torch.isfinite(torch.tensor(values)).all() for values in (
        result.exact_k_pass_nll_by_offset,
        result.live_feedback_nll_by_offset,
        result.standard_k1_nll_by_offset,
    ))
    # Offset zero and the first processed token still share the exact handoff;
    # later tokens exercise the collapsed Live Feedback stream.
    assert abs(result.exact_k_pass_nll_by_offset[0] - result.live_feedback_nll_by_offset[0]) < 1e-7
    assert abs(result.exact_k_pass_nll_by_offset[1] - result.live_feedback_nll_by_offset[1]) < 1e-7
    assert max(result.hidden_delta_rms_by_step[2:]) > 1e-5
    assert all(item.live_feedback_nll >= 0.0 for item in result.horizons)
