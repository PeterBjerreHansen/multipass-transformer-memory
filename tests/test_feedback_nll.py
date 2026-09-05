from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from conftest import micro_config
from test_evaluation_contract import Rows, make_model
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.evaluation import feedback
from tiny_mistral_mptt.evaluation.nll import evaluate_nll
from tiny_mistral_mptt.inference import (
    live_feedback_decode_step,
    prefill_live_feedback,
)
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.variants.memory_attention import MemoryAttentionVariant
from tiny_mistral_mptt.variants.recurrent_memory import RecurrentMemoryVariant


@pytest.mark.parametrize("kind", ["memory_add", "memory_token", "projected_residual", "recirculation",
                                  "dense", "strided", "dense_and_strided",
                                  "hybrid_projected_residual", "hybrid_recirculation"])
def test_bos_feedback_matches_observed_token_reference_and_model_owned_targets(kind, monkeypatch):
    if kind.startswith("hybrid_"):
        model = build_variant("dense_and_strided_memory_attention",
            MistralForCausalLM(micro_config(), attention_backend="reference"),
            memory_layers=[1], recurrent_layers=[1],
            recurrent_merger=kind.removeprefix("hybrid_"),
            memory_dense_window=2, memory_sparse_window=2, memory_sparse_stride=2,
        )
        with torch.no_grad():
            model.writer.proj.weight.add_(0.05 * torch.randn_like(model.writer.proj.weight))
            model.memory_readers["1"].o_proj.weight.copy_(torch.eye(model.config.hidden_size))
            if model.recurrent_merger == "projected_residual":
                model.memory_mergers["1"].projection.weight.fill_(0.03)
    elif kind == "dense_and_strided":
        model = MemoryAttentionVariant(
            MistralForCausalLM(micro_config(), attention_backend="reference"),
            memory_pattern="dense_and_strided",
            memory_dense_window=2, memory_sparse_window=2, memory_sparse_stride=2,
        )
    elif kind in {"dense", "strided"}:
        model = MemoryAttentionVariant(MistralForCausalLM(micro_config(), attention_backend="reference"),
                            memory_write_mode=kind, memory_write_stride=2, memory_window=3)
    else:
        model = make_model(kind)
    data = Rows(controls=kind == "memory_token")
    data.rows[:, 0] = 21  # The first data token is deliberately not BOS.
    prompts = []
    original = feedback.prefill_live_feedback
    def capture(model, ids, **kwargs):
        prompts.append((ids.tolist(), kwargs))
        return original(model, ids, **kwargs)
    monkeypatch.setattr(feedback, "prefill_live_feedback", capture)
    result = evaluate_nll(model, data, device="cpu", forward_mode="feedback")
    assert prompts == [([[1]], {"passes": 1, "decode_mode": "feedback"})] * 2
    assert model.training
    losses, aligned_losses, counts = [], [], []
    with torch.no_grad():
        for index in range(2):
            tokens = torch.cat((torch.tensor([[1]]), data.rows[index:index + 1]), dim=1)
            state = prefill_live_feedback(
                model, tokens[:, :1], passes=1, decode_mode="feedback"
            )
            logits = [state.next_token_logits]
            for position in range(1, tokens.shape[1] - 1):
                state = live_feedback_decode_step(
                    model, state, tokens[:, position : position + 1]
                )
                logits.append(state.next_token_logits)
            labels = model.build_lm_labels(tokens)[:, :-1]
            loss = F.cross_entropy(torch.stack(logits, dim=1).flatten(0, 1), labels.flatten(), reduction="none")
            losses.append(float(loss.sum()))
            aligned_losses.append(float(loss[1:].sum()))
            counts.append(int(labels.ne(-100).sum()))
    assert result.nll == pytest.approx(sum(losses) / sum(counts), abs=1e-6)
    assert result.aligned_nll == pytest.approx(sum(aligned_losses) / (sum(counts) - 2), abs=1e-6)
    assert result.predicted_tokens == sum(counts)
    assert result.aligned_predicted_tokens == sum(counts) - 2
    assert result.predicted_tokens_by_source == dict(zip(("a", "b"), counts))
    assert result.evaluation["policy"]["prompt_kind"] == "bos"
    assert result.evaluation["policy"]["synthetic_bos_counts_as_physical_token"] is True
    assert result.evaluation["policy"]["memory_write_position_basis"] == (
        "zero_based_physical_sequence_position"
    )
    assert result.evaluation["policy"]["periodic_write_rule"] == (
        "(position + 1) % stride == 0"
    )
    assert result.evaluation["policy"]["synthetic_bos_shifts_periodic_write_phase"] is (
        kind == "strided"
    )


def test_full_2048_block_scores_all_targets_without_context_overflow(monkeypatch):
    model = RecurrentMemoryVariant(
        MistralForCausalLM(micro_config(max_position_embeddings=2048), attention_backend="reference"),
        memory_layers=[1], merger="projected_residual",
    )
    data = Rows()
    data.rows = (torch.arange(2048)[None, :] % 90) + 3
    data.sequence_length = 2048
    consumed = []
    original = feedback.live_feedback_decode_step
    def capture(model, state, token):
        consumed.append(state.next_position)
        return original(model, state, token)
    monkeypatch.setattr(feedback, "live_feedback_decode_step", capture)
    result = feedback.evaluate_feedback_nll(model, data, device="cpu", max_blocks=1)
    assert result.predicted_tokens == 2048 and result.aligned_predicted_tokens == 2047
    assert consumed == list(range(1, 2048))


def test_feedback_rejects_unsupported_requests_and_restores_mode_on_interrupt():
    with pytest.raises(ValueError, match="does not implement feedback"):
        feedback.evaluate_feedback_nll(make_model("vanilla"), Rows(), device="cpu")
    model, data = make_model(), Rows()
    data.manifest = SimpleNamespace(source_ids={"a": 0, "b": 1}, bos_token_id=9)
    with pytest.raises(ValueError, match="BOS token IDs differ"):
        feedback.evaluate_feedback_nll(model, data, device="cpu")
    data = Rows()
    data.sequence_length = 257
    with pytest.raises(ValueError, match="not cropped"):
        feedback.evaluate_feedback_nll(model, data, device="cpu")
    with pytest.raises(InterruptedError):
        feedback.evaluate_feedback_nll(model, Rows(), device="cpu", stop_requested=lambda: True)
    assert model.training
    with pytest.raises(ValueError, match="passes=1"):
        evaluate_nll(model, Rows(), device="cpu", forward_mode="feedback", passes=4)


@pytest.mark.parametrize("precision", [None, "bfloat16"])
def test_feedback_uses_shared_precision_context_and_restores_mode(monkeypatch, precision):
    from tiny_mistral_mptt.evaluation import common
    calls = []
    def cpu_test_autocast(device, dtype):
        calls.append(dtype)
        return torch.autocast("cpu", dtype=torch.bfloat16)
    monkeypatch.setattr(common, "autocast_context", cpu_test_autocast)
    model = make_model("projected_residual")
    result = feedback.evaluate_feedback_nll(model, Rows(), device="cpu", autocast_dtype=precision)
    assert calls == ([] if precision is None else [precision])
    assert result.evaluation["precision"]["autocast_dtype"] == precision
    assert model.training
