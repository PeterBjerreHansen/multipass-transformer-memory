"""Retired serialized names belong here, not in the active implementation tests."""

from dataclasses import asdict

import pytest
import torch

from conftest import micro_config
from test_experiment_checkpoint import _objects
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.config import ExperimentConfig, SUPPORTED_VARIANTS, canonical_variant_name
from tiny_mistral_mptt.data.packed_dataset import StatefulBlockSampler
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.training.checkpoint import (
    TrainState, load_checkpoint, load_model_weights, save_checkpoint,
)


COMBINED_ALIASES = [
    "bank_multiscale", "multiscale_memory_attention",
    "memory_attention_multiscale", "attention_dense_and_strided",
]


@pytest.mark.parametrize("legacy_name,current_name,fields", [
    ("bank", "memory_attention", {"memory_write_mode": "dense"}),
    *[(name, "dense_and_strided_memory_attention", {
        "memory_dense_window": 2, "memory_sparse_window": 3, "memory_sparse_stride": 4,
    }) for name in COMBINED_ALIASES],

])
def test_old_inputs_load_but_emit_current_names_and_identical_models(legacy_name, current_name, fields):
    config = ExperimentConfig(variant=legacy_name, memory_layers=[0, 1], max_unique_tokens=1, **fields)
    config.validate()
    assert config.variant == asdict(config)["variant"] == current_name
    assert canonical_variant_name(legacy_name) == "memory_attention"
    assert legacy_name not in SUPPORTED_VARIANTS
    assert current_name in SUPPORTED_VARIANTS

    def model(name):
        torch.manual_seed(42)
        backbone = MistralForCausalLM(micro_config(num_hidden_layers=2), attention_backend="reference")
        result = build_variant(name, backbone, memory_layers=[0, 1], **fields).eval()
        with torch.no_grad():
            for reader in result.memory_readers.values():
                reader.o_proj.weight.copy_(0.05 * torch.eye(result.config.hidden_size))
        return result

    restored, current = model(legacy_name), model(current_name)
    assert restored.variant_name == current.variant_name == current_name
    assert type(restored) is type(current)
    assert restored.state_dict().keys() == current.state_dict().keys()
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[key], atol=0, rtol=0)
    ids = torch.tensor([[1, 7, 3, 14, 9, 22]])
    with torch.no_grad():
        torch.testing.assert_close(
            restored.compute_passes(ids, passes=3).final.logits,
            current.compute_passes(ids, passes=3).final.logits, atol=0, rtol=0,
        )


@pytest.mark.parametrize("legacy_name", COMBINED_ALIASES)
@pytest.mark.parametrize("reverse", [False, True])
def test_serialized_aliases_preserve_weights_and_resume_but_not_changed_readers(tmp_path, legacy_name, reverse):
    model, optimizer = _objects()
    recorded_name, requested_name = legacy_name, "dense_and_strided_memory_attention"
    if reverse:
        recorded_name, requested_name = requested_name, recorded_name
    architecture = dict(
        memory_layers=[3, 7], memory_window=64, memory_dense_window=32,
        memory_sparse_window=32, memory_sparse_stride=32,
    )
    path = save_checkpoint(
        tmp_path / "combined.pt", model=model, optimizer=optimizer,
        sampler_state=StatefulBlockSampler(5, seed=3).state_dict(),
        train_state=TrainState(optimizer_steps=7), data_manifest_sha256="same",
        experiment_config={"variant": recorded_name, **architecture},
    )
    replacement, replacement_optimizer = _objects()
    expected = {"variant": requested_name, **architecture}
    load_model_weights(path, model=replacement, expected_experiment_config=expected)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(replacement.state_dict()[key], value, atol=0, rtol=0)
    state, _ = load_checkpoint(
        path, model=replacement, optimizer=replacement_optimizer,
        expected_manifest_sha256="same", expected_experiment_config=expected,
    )
    assert state.optimizer_steps == 7
    changed = {**expected, "memory_layers": [4, 7]}
    with pytest.raises(ValueError, match="memory_layers"):
        load_model_weights(path, model=replacement, expected_experiment_config=changed)
    with pytest.raises(ValueError, match="memory_layers"):
        load_checkpoint(
            path, model=replacement, optimizer=replacement_optimizer,
            expected_manifest_sha256="same", expected_experiment_config=changed,
        )


@pytest.mark.parametrize("legacy_name", [
    "memory_bank32", "dense_memory_bank", "sparse_memory_bank", "memory_add_sparse_bank",
])
def test_removed_unstructured_aliases_still_fail(legacy_name):
    backbone = MistralForCausalLM(micro_config(), attention_backend="reference")
    with pytest.raises(ValueError, match="unknown variant"):
        build_variant(legacy_name, backbone)


@pytest.mark.parametrize("name", [
    "memory_attention_add_hybrid", "recirculation_strided_memory_attention",
    "memory_attention_recirculation_hybrid", "bank_add_hybrid",
    "bank_recirculation_hybrid", "tape_add_hybrid", "tape_recirculation_hybrid",
])
def test_deleted_hybrid_names_fail_at_every_loading_boundary(tmp_path, name):
    with pytest.raises(ValueError, match="removed"):
        ExperimentConfig.from_dict({"variant": name, "memory_write_mode": "dense"})
    with pytest.raises(ValueError, match="removed"):
        build_variant(name, MistralForCausalLM(micro_config(), attention_backend="reference"))
    model, optimizer = _objects()
    path = save_checkpoint(
        tmp_path / "removed.pt", model=model, optimizer=optimizer,
        sampler_state=StatefulBlockSampler(5, seed=3).state_dict(),
        train_state=TrainState(), data_manifest_sha256="same",
        experiment_config={"variant": name, "memory_write_mode": "dense"},
    )
    with pytest.raises(ValueError, match="removed"):
        load_checkpoint(path, model=model, optimizer=optimizer, expected_manifest_sha256="same")
    with pytest.raises(ValueError, match="removed"):
        load_model_weights(path, model=model, expected_experiment_config={
            "variant": "dense_memory_attention", "recurrent_merger": "projected_residual",
            "recurrent_layers": [0],
        })


@pytest.mark.parametrize("name", ["strided_attention", "sparse_swa"])
def test_old_non_memory_inputs_emit_strided_self_attention(name):
    fields = {"sparse_attention_stride": 2, "sparse_attention_window": 2}
    config = ExperimentConfig.from_dict({"variant": name, **fields})
    assert config.variant == "strided_self_attention"
    model = build_variant(name, MistralForCausalLM(micro_config(), attention_backend="reference"), **fields)
    assert model.variant_name == "strided_self_attention"
    assert not list(model.added_parameters())
