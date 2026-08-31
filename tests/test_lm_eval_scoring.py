from types import SimpleNamespace

import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.evaluation.lm_eval_adapter import score_token_continuation
from tiny_mistral_mptt.evaluation.lm_eval_adapter import (
    _TokenizerFacade,
    generate_recurrent,
    make_lm_eval_adapter,
    score_token_continuation_recurrent,
)
from tiny_mistral_mptt.inference import (
    paper_recirculation_decode_step,
    prefill_paper_recirculation,
    prefill_recurrent,
    recurrent_decode_step,
)
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant


class NextIdModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, ids, use_cache=False):
        batch, length = ids.shape
        logits = torch.full((batch, length, self.vocab_size), -20.0)
        next_ids = (ids + 1) % self.vocab_size
        logits.scatter_(-1, next_ids.unsqueeze(-1), 20.0)
        return SimpleNamespace(logits=logits)


def test_token_continuation_scores_only_continuation_and_greedy_contract():
    model = NextIdModel()
    score, greedy = score_token_continuation(
        model,
        device="cpu",
        max_length=16,
        context_enc=[10, 11],
        continuation_enc=[12, 13],
    )
    assert greedy is True
    assert score > -1e-5

    _score, greedy = score_token_continuation(
        model,
        device="cpu",
        max_length=16,
        context_enc=[10, 11],
        continuation_enc=[12, 14],
    )
    assert greedy is False


def test_token_continuation_left_truncation_keeps_requested_targets():
    model = NextIdModel()
    score, greedy = score_token_continuation(
        model,
        device="cpu",
        max_length=3,
        context_enc=[7, 8, 9],
        continuation_enc=[10, 11],
    )
    assert greedy is True
    assert score > -1e-5


def test_harness_rejects_feedback_for_a_vanilla_model_before_tokenizer_loading():
    with pytest.raises(ValueError, match="vanilla models do not implement feedback"):
        make_lm_eval_adapter(
            NextIdModel(),
            tokenizer_path="does-not-exist.json",
            device="cpu",
            prefill_passes=1,
            decode_mode="feedback",
        )


def test_harness_tokenizer_disables_training_padding_and_truncation(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(length=8, pad_id=1, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=1)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))

    facade = _TokenizerFacade(path)
    assert facade.encode("hello hello") == [2, 2]


def _make_memory_model():
    torch.manual_seed(444)
    backbone = MistralForCausalLM(
        micro_config(sliding_window=4), attention_backend="reference"
    )
    model = MemoryAddVariant(backbone).eval()
    with torch.no_grad():
        model.memory_projection.weight.copy_(0.05 * torch.eye(model.config.hidden_size))
    return model


def _make_recirculation_model():
    torch.manual_seed(445)
    return RecirculationVariant(
        MistralForCausalLM(
            micro_config(sliding_window=4), attention_backend="reference"
        ),
        source_layer=1,
        destination_layer=0,
        mode="adaptive",
    ).eval()


def test_recurrent_task_scoring_uses_one_collapsed_stream(monkeypatch):
    model = _make_memory_model()

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("recurrent task scoring must not call public forward")

    monkeypatch.setattr(model, "forward", forbidden_forward)
    score, greedy = score_token_continuation_recurrent(
        model,
        device="cpu",
        max_length=16,
        prefill_passes=2,
        decode_mode="feedback",
        context_enc=[1, 7, 3, 14],
        continuation_enc=[22, 9, 31],
    )

    state = prefill_recurrent(
        model,
        torch.tensor([[1, 7, 3, 14]], dtype=torch.long),
        passes=2,
        decode_mode="feedback",
    )
    expected_score = 0.0
    expected_greedy = True
    targets = [22, 9, 31]
    for index, token_id in enumerate(targets):
        logits = state.next_token_logits.float()
        expected_score += float(torch.log_softmax(logits, dim=-1)[0, token_id])
        expected_greedy = expected_greedy and bool(
            torch.argmax(logits, dim=-1).item() == token_id
        )
        if index + 1 < len(targets):
            state = recurrent_decode_step(
                model,
                state,
                torch.tensor([[token_id]], dtype=torch.long),
            )

    assert abs(score - expected_score) < 1e-6
    assert greedy is expected_greedy


def test_recurrent_task_generation_matches_incremental_greedy_decode(monkeypatch):
    model = _make_memory_model()

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("recurrent generation must not call public forward")

    monkeypatch.setattr(model, "forward", forbidden_forward)
    prompt = torch.tensor([[1, 7, 3, 14]], dtype=torch.long)
    generated = generate_recurrent(
        model,
        prompt,
        3,
        prefill_passes=2,
        decode_mode="feedback",
        temperature=0.0,
    )

    state = prefill_recurrent(
        model, prompt, passes=2, decode_mode="feedback"
    )
    expected = prompt.clone()
    for step in range(3):
        token = torch.argmax(state.next_token_logits, dim=-1, keepdim=True)
        expected = torch.cat((expected, token), dim=1)
        if step < 2:
            state = recurrent_decode_step(model, state, token)

    torch.testing.assert_close(generated, expected)


def test_paper_recirculation_generation_uses_replayed_cache() -> None:
    model = _make_recirculation_model()
    prompt = torch.tensor([[1, 7, 3, 14]], dtype=torch.long)

    generated = generate_recurrent(
        model,
        prompt,
        3,
        prefill_passes=1,
        decode_mode="paper_recirculation",
        temperature=0.0,
    )

    state = prefill_paper_recirculation(model, prompt)
    expected = prompt.clone()
    for step in range(3):
        token = torch.argmax(state.next_token_logits, dim=-1, keepdim=True)
        expected = torch.cat((expected, token), dim=1)
        if step < 2:
            state = paper_recirculation_decode_step(model, state, token)

    torch.testing.assert_close(generated, expected)
    with pytest.raises(ValueError, match="no prompt K axis"):
        generate_recurrent(
            model,
            prompt,
            1,
            prefill_passes=2,
            decode_mode="paper_recirculation",
        )
