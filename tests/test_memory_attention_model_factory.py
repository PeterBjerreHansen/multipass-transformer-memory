import pytest

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.config import ExperimentConfig, canonical_variant_name
from tiny_mistral_mptt.variants import (
    MemoryAttentionVariant,
    MemoryAttentionRecurrentHybridVariant,
    StridedSelfAttentionVariant,
    SwaTransformerVariant,
)
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant


def backbone():
    return MistralForCausalLM(micro_config(num_hidden_layers=1), attention_backend="reference")


def test_factory_exposes_only_clean_memory_names_and_policies():
    dense = build_variant("memory_attention", backbone(), memory_write_mode="dense")
    assert isinstance(dense, MemoryAttentionVariant)
    assert dense.memory_write_mode == "dense"

    hybrid = build_variant(
        "memory_token_attention",
        backbone(),
        memory_write_mode="memory_token",
        memory_write_stride=8,
        memory_token_visibility="write_only",
        recurrent_merger="projected_residual", recurrent_layers=[0],
    )
    assert isinstance(hybrid, MemoryAttentionRecurrentHybridVariant)


def test_memory_attention_names_are_public_model_names():
    assert canonical_variant_name("memory_attention") == "memory_attention"
    config = ExperimentConfig(
        variant="memory_attention",
        memory_write_mode="dense",
        max_unique_tokens=1,
    )
    config.validate()
    model = build_variant("memory_attention", backbone(), memory_write_mode="dense")
    assert isinstance(model, MemoryAttentionVariant)
    assert model.variant_name == "memory_attention"

    dense_and_strided = build_variant(
        "dense_and_strided_memory_attention",
        MistralForCausalLM(
            micro_config(num_hidden_layers=3), attention_backend="reference"
        ),
        memory_dense_window=4,
        memory_sparse_window=3,
        memory_sparse_stride=8,
        memory_layers=[1, 2],
    )
    assert isinstance(dense_and_strided, MemoryAttentionVariant)
    assert dense_and_strided.variant_name == "dense_and_strided_memory_attention"


def test_dense_and_strided_name_preserves_retention():
    name = "dense_and_strided_memory_attention"
    config = ExperimentConfig(
        variant=name, memory_layers=[0, 1], memory_dense_window=2,
        memory_sparse_window=3, memory_sparse_stride=4, max_unique_tokens=1,
    )
    config.validate()
    assert canonical_variant_name(name) == "memory_attention"
    assert config.memory_window == 5
    assert config.memory_write_mode is None
    model = build_variant(
        name, MistralForCausalLM(micro_config(num_hidden_layers=2), attention_backend="reference"),
        memory_layers=config.memory_layers, memory_dense_window=config.memory_dense_window,
        memory_sparse_window=config.memory_sparse_window, memory_sparse_stride=config.memory_sparse_stride,
    )
    assert isinstance(model, MemoryAttentionVariant)
    assert model.variant_name == name
    assert model.memory_window == 5
    assert model.memory_layers == (0, 1)


def test_public_model_names_select_public_implementations():
    baseline = build_variant("swa_transformer", backbone())
    assert isinstance(baseline, SwaTransformerVariant)
    strided = build_variant(
        "strided_self_attention",
        MistralForCausalLM(micro_config(num_hidden_layers=1), attention_backend="reference"),
        sparse_attention_stride=8,
        sparse_attention_window=8,
    )
    assert isinstance(strided, StridedSelfAttentionVariant)
    memory = build_variant(
        "strided_memory_attention",
        backbone(),
        memory_write_mode="strided",
        memory_write_stride=8,
    )
    assert isinstance(memory, MemoryAttentionVariant)
    multi = build_variant(
        "dense_and_strided_memory_attention",
        MistralForCausalLM(micro_config(num_hidden_layers=2), attention_backend="reference"),
        memory_dense_window=2,
        memory_sparse_window=2,
        memory_sparse_stride=4,
    )
    assert isinstance(multi, MemoryAttentionVariant)
    hybrid = build_variant(
        "strided_memory_attention",
        MistralForCausalLM(micro_config(num_hidden_layers=2), attention_backend="reference"),
        memory_write_mode="strided",
        memory_write_stride=8,
        recurrent_merger="recirculation", recurrent_layers=[0],
    )
    assert isinstance(hybrid, MemoryAttentionRecurrentHybridVariant)


def test_factory_selects_only_requested_memory_layers_and_defaults_to_rope():
    multi_layer = MistralForCausalLM(
        micro_config(num_hidden_layers=3), attention_backend="reference"
    )
    model = build_variant(
        "memory_attention",
        multi_layer,
        memory_write_mode="periodic",
        memory_write_stride=8,
        memory_layers=[0, 2],
    )
    assert model.memory_layers == (0, 2)
    assert list(model.memory_readers) == ["0", "2"]
    assert all(reader.position_encoding == "rope" for reader in model.memory_readers.values())

    with pytest.raises(ValueError, match="memory_layers"):
        build_variant(
            "memory_attention",
            MistralForCausalLM(
                micro_config(num_hidden_layers=2), attention_backend="reference"
            ),
            memory_write_mode="periodic",
            memory_write_stride=8,
            memory_layers=[2],
        )


def test_factory_builds_dense_and_strided_memory_without_a_write_policy_axis():
    model = build_variant(
        "dense_and_strided_memory_attention",
        MistralForCausalLM(
            micro_config(num_hidden_layers=3), attention_backend="reference"
        ),
        memory_dense_window=4,
        memory_sparse_window=3,
        memory_sparse_stride=8,
        memory_layers=[1, 2],
    )
    assert isinstance(model, MemoryAttentionVariant)
    assert model.memory_window == 7
    assert model.memory_layers == (1, 2)
    assert model.memory_write_mode == "dense"

    with pytest.raises(ValueError, match="conflicts with memory_write_mode"):
        build_variant(
            "dense_and_strided_memory_attention",
            backbone(),
            memory_write_mode="dense",
        )



def test_factory_requires_memory_token_visibility_explicitly():
    with pytest.raises(ValueError, match="memory_token_visibility"):
        build_variant(
            "memory_attention",
            backbone(),
            memory_write_mode="memory_token",
            memory_write_stride=8,
        )


def test_factory_builds_recirculation_with_explicit_layer_contract():
    two_layer_backbone = MistralForCausalLM(
        micro_config(num_hidden_layers=2), attention_backend="reference"
    )
    model = build_variant(
        "recirculation",
        two_layer_backbone,
        recirculation_source_layer=1,
        recirculation_destination_layer=0,
    )
    assert isinstance(model, RecirculationVariant)

    adaptive = build_variant(
        "recirculation",
        MistralForCausalLM(
            micro_config(num_hidden_layers=2), attention_backend="reference"
        ),
        recirculation_source_layer=1,
        recirculation_destination_layer=0,
        recirculation_mode="adaptive",
    )
    assert isinstance(adaptive, RecirculationVariant)
    assert adaptive.mode == "adaptive"
    assert list(adaptive.added_parameters())


def test_factory_builds_optional_late_recurrent_memory_hybrid():
    model = build_variant(
        "memory_attention",
        MistralForCausalLM(
            micro_config(num_hidden_layers=3), attention_backend="reference"
        ),
        memory_write_mode="periodic",
        memory_write_stride=8,
        memory_layers=[1],
        recurrent_merger="recirculation",
        recurrent_layers=[0, 2],
    )
    assert isinstance(model, MemoryAttentionRecurrentHybridVariant)
    assert model.recurrent_merger == "recirculation"
    assert model.recurrent_layers == (0, 2)
    assert model.memory_layers == (1,)


@pytest.mark.parametrize("name,pattern,mode,fields", [
    ("dense_memory_attention", "dense", "dense", {}),
    ("strided_memory_attention", "strided", "strided", {"memory_write_stride": 2}),
    ("dense_and_strided_memory_attention", "dense_and_strided", None,
     {"memory_dense_window": 2, "memory_sparse_window": 2, "memory_sparse_stride": 2}),
    ("memory_token_attention", "dense", "memory_token",
     {"memory_write_stride": 2, "memory_token_visibility": "write_only"}),
])
def test_descriptive_names_are_presets_of_one_implementation(name, pattern, mode, fields):
    config = ExperimentConfig(variant=name, max_unique_tokens=1, **fields)
    config.validate()
    assert canonical_variant_name(name) == "memory_attention"
    assert (config.memory_pattern, config.memory_write_mode) == (pattern, mode)
    model = build_variant(name, backbone(), **fields)
    generic = build_variant("memory_attention", backbone(), memory_pattern=pattern,
                            memory_write_mode=mode, **fields)
    assert type(model) is type(generic) is MemoryAttentionVariant
    assert model.memory_pattern == generic.memory_pattern == pattern
    assert model.variant_name == name


@pytest.mark.parametrize("fields", [
    {"variant": "dense_memory_attention", "memory_pattern": "strided"},
    {"variant": "dense_memory_attention", "memory_write_mode": "memory_token"},
    {"variant": "strided_memory_attention", "memory_write_mode": "dense"},
    {"variant": "dense_and_strided_memory_attention", "memory_write_mode": "dense"},
    {"variant": "memory_attention", "memory_pattern": "strided", "memory_write_mode": "dense"},
    {"variant": "memory_attention", "memory_pattern": "unknown"},
])
def test_conflicting_presets_fail_in_config_and_factory(fields):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(fields)
    fields = dict(fields)
    name = fields.pop("variant")
    with pytest.raises(ValueError):
        build_variant(name, backbone(), **fields)


@pytest.mark.parametrize("fields", [
    {"recurrent_layers": [0]},
    {"recurrent_merger": "projected_residual"},
    {"recurrent_merger": "unknown", "recurrent_layers": [0]},
    {"recurrent_merger": "recirculation", "recurrent_layers": [2]},
    {"recirculation_source_layer": 1, "recirculation_destination_layer": 0},
])
def test_factory_rejects_incomplete_or_obsolete_hybrid_settings(fields):
    with pytest.raises(ValueError):
        build_variant("dense_memory_attention", backbone(), **fields)
