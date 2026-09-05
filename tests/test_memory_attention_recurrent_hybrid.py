import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.feedback import HybridFeedbackState, HybridPassSource
from tiny_mistral_mptt.inference import (
    exact_decode_step,
    live_feedback_decode_step,
    live_feedback_from_exact,
    prefill_exact_k_pass,
    prefill_live_feedback,
)
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.recurrent_memory import RecurrentMemoryVariant
from tiny_mistral_mptt.variants.memory_attention_recurrent_hybrid import (
    MemoryAttentionRecurrentHybridVariant,
)


def make_backbone(seed: int = 123) -> MistralForCausalLM:
    torch.manual_seed(seed)
    return MistralForCausalLM(
        micro_config(num_hidden_layers=3, sliding_window=4),
        attention_backend="reference",
    )


def make_hybrid(*, mode: str = "periodic", merger="projected_residual", layers=(0,)) -> MemoryAttentionRecurrentHybridVariant:
    model = MemoryAttentionRecurrentHybridVariant(
        make_backbone(),
        recurrent_merger=merger,
        recurrent_layers=list(layers),
        memory_window=3,
        memory_write_mode=mode,
        memory_write_stride=2,
        memory_token_visibility="visible",
        memory_layers=[1],
        initialization_seed=991,
    )
    return model


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
def test_zero_memory_readers_reduce_hybrid_to_same_recurrent_memory(merger):
    hybrid = make_hybrid(merger=merger).eval()
    recirculation = RecurrentMemoryVariant(
        make_backbone(),
        merger=merger,
        memory_layers=[0],
        initialization_seed=991,
    ).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        hybrid.writer.proj.weight.add_(0.02 * torch.randn_like(hybrid.writer.proj.weight))
        if merger == "projected_residual":
            hybrid.memory_mergers["0"].projection.weight.fill_(0.02)
        recirculation.writer.load_state_dict(hybrid.writer.state_dict())
        recirculation.memory_mergers.load_state_dict(hybrid.memory_mergers.state_dict())

    with torch.no_grad():
        hybrid_output = hybrid.compute_passes(ids, passes=3)
        recirculation_output = recirculation.compute_passes(ids, passes=3)

    for actual, expected in zip(
        hybrid_output.passes, recirculation_output.passes, strict=True
    ):
        torch.testing.assert_close(
            actual.hidden_states, expected.hidden_states, atol=2e-6, rtol=2e-6
        )


def test_both_channels_use_the_same_late_normalized_source():
    model = make_hybrid().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        run = model._run_first_state(ids)

    assert isinstance(run.feedback_source, HybridPassSource)
    assert run.feedback_source.recurrent_hidden.shape == run.hidden_states.shape
    torch.testing.assert_close(
        run.feedback_source.memory_attention_hidden, run.hidden_states, atol=0, rtol=0
    )
    torch.testing.assert_close(
        run.feedback_source.recurrent_hidden, run.feedback_source.memory_attention_hidden, atol=0, rtol=0
    )


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
def test_phase_a_trains_attention_reader_and_recurrent_merger(merger):
    model = make_hybrid(merger=merger)
    configure_phase(model, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = model.compute_loss(
        ids, phase="A", passes=2, loss_weights=[0.0, 1.0]
    )
    output.loss.backward()

    reader_grad = model.memory_readers["1"].o_proj.weight.grad
    assert reader_grad is not None and reader_grad.abs().sum() > 0
    module = model.memory_mergers["0"]
    weights = module.projection.weight if merger == "projected_residual" else module.controller.output.weight
    assert weights.grad is not None and weights.grad.abs().sum() > 0
    added = {id(parameter) for parameter in model.added_parameters()}
    for parameter in model.parameters():
        if id(parameter) not in added:
            assert parameter.grad is None


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
@pytest.mark.parametrize("layers", [(0,), (1,), (0, 1, 2)])
@pytest.mark.parametrize("mode", ["periodic", "memory_token"])
def test_cached_exact_k_pass_matches_full_prefix(mode, merger, layers):
    model = make_hybrid(mode=mode, merger=merger, layers=layers).eval()
    with torch.no_grad():
        model.writer.proj.weight.add_(0.03 * torch.randn_like(model.writer.proj.weight))
        if merger == "projected_residual":
            for module in model.memory_mergers.values():
                module.projection.weight.copy_(0.05 * torch.eye(model.config.hidden_size))
        model.memory_readers["1"].o_proj.weight.copy_(
            torch.eye(model.config.hidden_size)
        )
    if mode == "memory_token":
        vocab = model.config.vocab_size
        ids = torch.tensor([[1, 2, vocab, 3, 4, vocab, 5]])
    else:
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])

    prompt_length = 4
    with torch.no_grad():
        state = prefill_exact_k_pass(model, ids[:, :prompt_length], passes=2)
        for position in range(prompt_length, ids.shape[1] + 1):
            prefix = ids[:, :position]
            full = model.compute_passes(prefix, passes=2)
            expected = model.backbone.lm_head(
                model.prediction_hidden_after_sequence(
                    full.final.hidden_states, prefix
                )
            ).float()[:, -1, :]
            torch.testing.assert_close(
                state.next_token_logits, expected, atol=8e-5, rtol=8e-5
            )
            if position < ids.shape[1]:
                state = exact_decode_step(
                    model, state, ids[:, position : position + 1]
                )

    if mode == "memory_token":
        assert isinstance(state.streams[-1].feedback_memory, HybridFeedbackState)


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
@pytest.mark.parametrize("pattern", ["dense", "strided", "dense_and_strided", "memory_token"])
def test_hybrid_k1_conversion_and_feedback_keep_emitted_state(merger, pattern):
    fields = {"memory_pattern": pattern}
    if pattern == "dense_and_strided":
        fields.update(memory_dense_window=2, memory_sparse_window=2, memory_sparse_stride=2)
    elif pattern == "strided":
        fields["memory_write_stride"] = 2
    elif pattern == "memory_token":
        fields = {"memory_write_mode": "memory_token", "memory_write_stride": 2,
                  "memory_token_visibility": "write_only"}
    model = build_variant("memory_attention", make_backbone(), memory_layers=[1],
                          recurrent_merger=merger, recurrent_layers=[1], **fields).eval()
    with torch.no_grad():
        model.writer.proj.weight.add_(0.04 * torch.randn_like(model.writer.proj.weight))
        model.memory_readers["1"].o_proj.weight.copy_(torch.eye(model.config.hidden_size))
        if merger == "projected_residual":
            model.memory_mergers["1"].projection.weight.fill_(0.03)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        if pattern == "memory_token":
            ids = torch.tensor([[1, 2, model.memory_token_id, 3, 4, model.memory_token_id, 5]])
        state = prefill_exact_k_pass(model, ids[:, :1], passes=1)
        for position in range(1, ids.shape[1] - 1):
            state = exact_decode_step(model, state, ids[:, position:position + 1])
            converted = live_feedback_from_exact(state, decode_mode="feedback")
            fresh = prefill_live_feedback(
                model,
                ids[:, :position + 1],
                passes=1,
                decode_mode="feedback",
            )
            torch.testing.assert_close(
                converted.feedback_memory.recurrent_memory, fresh.feedback_memory.recurrent_memory,
                atol=8e-5, rtol=8e-5,
            )
            next_token = ids[:, position + 1:position + 2]
            torch.testing.assert_close(
                live_feedback_decode_step(
                    model, converted, next_token
                ).next_token_logits,
                live_feedback_decode_step(model, fresh, next_token).next_token_logits,
                atol=8e-5,
                rtol=8e-5,
            )


def test_overlapping_readers_apply_attention_then_recurrence_before_mlp(monkeypatch):
    model = make_hybrid(layers=(1,)).eval()
    observed = {}

    def attention_delta(reader, hidden, memory, **kwargs):
        observed["after_attention"] = hidden + 0.25
        return torch.full_like(hidden, 0.25)

    def merger(destination, memory):
        torch.testing.assert_close(destination, observed["after_attention"], atol=0, rtol=0)
        observed["after_recurrence"] = destination + 0.5
        return observed["after_recurrence"]

    monkeypatch.setattr(model, "_full_memory_delta", attention_delta)
    monkeypatch.setattr(model.memory_mergers["1"], "forward", merger)
    inputs = []
    handle = model.backbone.model.layers[1].post_attention_layernorm.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone())
    )
    ids = torch.tensor([[1, 2, 3, 4]])
    try:
        with torch.no_grad():
            model._run_feedback_hidden(ids, model.input_embeddings(ids),
                                       torch.randn(1, 4, model.config.hidden_size))
    finally:
        handle.remove()
    expected = observed["after_recurrence"].clone()
    expected[:, 0] = observed["after_attention"][:, 0]  # no preceding token
    torch.testing.assert_close(inputs[0], expected, atol=0, rtol=0)
    added = list(model.added_parameters())
    assert len(added) == len({id(parameter) for parameter in added})
