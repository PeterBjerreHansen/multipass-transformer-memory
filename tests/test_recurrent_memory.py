import copy

import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.inference.multipass import prefill_exact, exact_decode_step
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.memory_attention import MemoryAttentionWriter
from tiny_mistral_mptt.variants.memory_modules import MemoryWriter
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant
from tiny_mistral_mptt.variants.recurrent_memory import RecurrentMemoryVariant


def make_variant(merger="projected_residual"):
    torch.manual_seed(30)
    backbone = MistralForCausalLM(micro_config(), attention_backend="reference")
    return RecurrentMemoryVariant(backbone, memory_layers=[0], merger=merger)


def activate(variant):
    with torch.no_grad():
        for merger in variant.memory_mergers.values():
            if hasattr(merger, "projection"):
                merger.projection.weight.normal_(std=0.1)


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
def test_late_writer_is_shared_and_uses_final_normalized_state(merger):
    variant = make_variant(merger).eval()
    assert type(variant.writer) is MemoryWriter is MemoryAttentionWriter
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    seen = []
    handle = variant.writer.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    with torch.no_grad():
        expected = variant.backbone.model(ids, use_cache=False).last_hidden_state
        variant.compute_passes(ids, passes=2)
    handle.remove()
    torch.testing.assert_close(seen[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(variant.writer(expected), expected, atol=0, rtol=0)


def test_projected_residual_starts_as_exact_vanilla_at_every_pass():
    variant = make_variant().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        expected = variant.backbone(ids, use_cache=False).logits
        result = variant.compute_passes(ids, passes=4)
    for run in result.passes:
        torch.testing.assert_close(run.logits, expected, atol=0, rtol=0)


def test_recirculation_merger_reuses_the_existing_adaptive_rule():
    variant = make_variant("recirculation")
    old = RecirculationVariant(copy.deepcopy(variant.backbone), source_layer=1, destination_layer=0, mode="adaptive")
    merger = variant.memory_mergers["0"]
    merger.controller.load_state_dict(old.adaptive_controller.state_dict())
    memory, destination = torch.randn(2, 4, 32), torch.randn(2, 4, 32)
    torch.testing.assert_close(merger(destination, memory), old._mix(memory, destination), atol=0, rtol=0)


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
def test_feedback_is_strictly_previous_token_and_position_zero_is_untouched(merger):
    variant = make_variant(merger).eval()
    activate(variant)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    embeddings = variant.input_embeddings(ids)
    previous = torch.randn_like(embeddings)
    changed = previous.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 4
    with torch.no_grad():
        first = variant._run_feedback_hidden(ids, embeddings, previous)
        second = variant._run_feedback_hidden(ids, embeddings, changed)
        vanilla = variant.backbone.model(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        memory, valid = variant._parallel_memory(previous)
    torch.testing.assert_close(memory[:, 1:], variant.writer(previous)[:, :-1], atol=0, rtol=0)
    assert not valid[:, 0].any() and valid[:, 1:].all()
    torch.testing.assert_close(first[:, 0], vanilla[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(first[:, :3], second[:, :3], atol=0, rtol=0)
    assert not torch.equal(first[:, 3:], second[:, 3:])


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
def test_frozen_training_can_learn_writer_without_changing_backbone(merger):
    variant = make_variant(merger)
    configure_phase(variant, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    original = {name: value.clone() for name, value in variant.backbone.state_dict().items()}
    optimizer = torch.optim.SGD(variant.added_parameters(), lr=0.1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        variant.compute_loss(ids, phase="A", passes=2, loss_weights=[0, 1]).loss.backward()
        if merger == "projected_residual" and step == 0:
            assert variant.memory_mergers["0"].projection.weight.grad.abs().sum() > 0
        if step == 1:
            assert variant.writer.proj.weight.grad.abs().sum() > 0
            assert torch.isfinite(variant.writer.proj.weight.grad).all()
        assert all(parameter.grad is None for parameter in variant.backbone.parameters())
        optimizer.step()
    for name, value in variant.backbone.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
def test_unfrozen_training_backpropagates_through_late_feedback(merger):
    variant = make_variant(merger)
    activate(variant)
    configure_phase(variant, "B")
    variant.compute_loss(torch.tensor([[1, 2, 3, 4, 5]]), phase="B", passes=3, loss_weights=[0, 0, 1]).loss.backward()
    assert variant.backbone.model.embed_tokens.weight.grad.abs().sum() > 0
    assert variant.writer.proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
@pytest.mark.parametrize("prompt_length", [1, 5])
def test_cached_fixed_k_matches_full_recomputation_after_reader_and_writer_updates(merger, prompt_length):
    variant = make_variant(merger).eval()
    activate(variant)
    with torch.no_grad():
        variant.writer.proj.weight.normal_(std=0.1)
        ids = torch.tensor([[1, 2, 3, 4, 5]])[:, :prompt_length]
        state = prefill_exact(variant, ids, passes=4)
        for value in [6, 7, 8]:
            token = torch.tensor([[value]])
            state = exact_decode_step(variant, state, token)
            ids = torch.cat((ids, token), dim=1)
            expected = variant.compute_passes(ids, passes=4).final.logits[:, -1, :]
            torch.testing.assert_close(state.next_token_logits, expected, atol=1e-6, rtol=1e-5)


def test_construction_is_rng_neutral_and_factory_preserves_dtype():
    torch.manual_seed(99)
    backbone = MistralForCausalLM(micro_config(), attention_backend="reference").double()
    before = torch.get_rng_state().clone()
    variant = build_variant("recurrent_memory", backbone, memory_layers=[0], recurrent_merger="projected_residual")
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
    assert all(parameter.dtype == torch.float64 for parameter in variant.parameters())


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
def test_config_and_state_dict_roundtrip(merger):
    config = ExperimentConfig.from_dict({"variant": "recurrent_memory", "memory_layers": [0], "memory_window": 1, "recurrent_merger": merger})
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    source, target = make_variant(merger), make_variant(merger)
    activate(source)
    target.load_state_dict(source.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    torch.testing.assert_close(source.compute_passes(ids, passes=3).final.logits, target.compute_passes(ids, passes=3).final.logits, atol=0, rtol=0)


@pytest.mark.parametrize("change", [
    {"memory_layers": "all"}, {"memory_layers": []}, {"memory_layers": [0, 0]},
    {"recurrent_merger": "unknown"}, {"memory_write_mode": "dense"},
    {"memory_position_encoding": "rope"}, {"memory_window": 2},
    {"training_forward": "recirculation_bptt"}, {"validation_forward": "paper_recirculation"},
    {"recirculation_source_layer": 1},
])
def test_config_rejects_inapplicable_modes(change):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"variant": "recurrent_memory", "memory_layers": [0], "memory_window": 1, "recurrent_merger": "projected_residual", **change})


def test_checkpoint_rejects_a_different_merger(tmp_path):
    from tiny_mistral_mptt.training.checkpoint import TrainState, save_checkpoint, load_model_weights

    model = make_variant()
    config = {"variant": "recurrent_memory", "memory_layers": [0], "memory_window": 1, "recurrent_merger": "projected_residual"}
    path = save_checkpoint(
        tmp_path / "checkpoint.pt", model=model,
        optimizer=torch.optim.AdamW(model.parameters()), sampler_state={},
        train_state=TrainState(), experiment_config=config, data_manifest_sha256="test",
    )
    load_model_weights(path, model=make_variant(), expected_experiment_config=config)
    with pytest.raises(ValueError, match="recurrent_merger"):
        load_model_weights(path, model=make_variant("recirculation"), expected_experiment_config={**config, "recurrent_merger": "recirculation"})


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
def test_shared_intervention_diagnostic_accepts_both_mergers(merger):
    from types import SimpleNamespace
    from tiny_mistral_mptt.evaluation.interventions import evaluate_memory_interventions

    class Dataset:
        sequence_length = 5
        manifest = SimpleNamespace(source_ids={"test": 0})

        def __len__(self):
            return 2

        def batch(self, indices, *, device):
            return torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], device=device)[indices]

        def source_id(self, index):
            return 0

    model = make_variant(merger).eval()
    activate(model)
    result = evaluate_memory_interventions(model, Dataset(), device="cpu", max_blocks=1)
    assert result["baseline_pass1"]["predicted_tokens"] == 4
    for condition in ("zero_memory", "mismatched_memory"):
        assert result["real_memory"]["nll"] != result[condition]["nll"]
