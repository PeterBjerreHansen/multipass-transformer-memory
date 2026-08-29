import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.evaluation.recurrent import evaluate_recurrent_continuation
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


def test_recurrent_eval_k1_uses_feedback_after_the_shared_prompt_prediction():
    result = evaluate_recurrent_continuation(
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
    assert result.exact_nll_by_offset[0] == result.recurrent_nll_by_offset[0]
    assert any(
        exact != recurrent
        for exact, recurrent in zip(
            result.exact_nll_by_offset[1:],
            result.recurrent_nll_by_offset[1:],
            strict=True,
        )
    )
    torch.testing.assert_close(
        torch.tensor(result.exact_nll_by_offset),
        torch.tensor(result.vanilla_nll_by_offset),
        atol=0,
        rtol=0,
    )
    assert max(result.hidden_delta_rms_by_step) > 1e-5


def test_recurrent_eval_k2_has_exact_initial_handoff_then_measures_drift():
    result = evaluate_recurrent_continuation(
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
    assert abs(result.exact_nll_by_offset[0] - result.recurrent_nll_by_offset[0]) < 1e-7
    assert abs(result.exact_nll_by_offset[1] - result.recurrent_nll_by_offset[1]) < 1e-7
    assert result.hidden_delta_rms_by_step[0] < 1e-6
    assert max(result.hidden_delta_rms_by_step[1:]) > 1e-5
    assert [item.horizon for item in result.horizons] == [1, 2, 4, 6]


def test_recurrent_eval_exercises_strided_memory_attention_after_prefill():
    result = evaluate_recurrent_continuation(
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
        result.exact_nll_by_offset,
        result.recurrent_nll_by_offset,
        result.vanilla_nll_by_offset,
    ))
    # Offset zero and the first processed token still share the exact handoff;
    # later tokens exercise the collapsed recurrent stream.
    assert abs(result.exact_nll_by_offset[0] - result.recurrent_nll_by_offset[0]) < 1e-7
    assert abs(result.exact_nll_by_offset[1] - result.recurrent_nll_by_offset[1]) < 1e-7
    assert max(result.hidden_delta_rms_by_step[2:]) > 1e-5
    assert all(item.recurrent_nll >= 0.0 for item in result.horizons)
