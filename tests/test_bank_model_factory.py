import pytest

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.config import ExperimentConfig, canonical_variant_name
from tiny_mistral_mptt.variants.bank import BankVariant
from tiny_mistral_mptt.variants.bank_add_hybrid import BankAddHybridVariant
from tiny_mistral_mptt.variants.bank_recirculation_hybrid import (
    BankRecirculationHybridVariant,
)
from tiny_mistral_mptt.variants.bank_multiscale import MultiscaleBankVariant
from tiny_mistral_mptt.variants import (
    MemoryAttentionVariant,
    MultiscaleMemoryAttentionVariant,
    RecirculationStridedMemoryAttentionVariant,
    StridedAttentionVariant,
    SwaTransformerVariant,
)
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant


def backbone():
    return MistralForCausalLM(micro_config(num_hidden_layers=1), attention_backend="reference")


def test_factory_exposes_only_clean_bank_names_and_policies():
    dense = build_variant("bank", backbone(), memory_write_mode="dense")
    assert isinstance(dense, BankVariant)
    assert dense.memory_write_mode == "dense"

    hybrid = build_variant(
        "bank_add_hybrid",
        backbone(),
        memory_write_mode="memory_token",
        memory_write_stride=8,
        memory_token_visibility="write_only",
    )
    assert isinstance(hybrid, BankAddHybridVariant)


def test_memory_attention_names_are_public_model_names():
    assert canonical_variant_name("memory_attention") == "bank"
    assert canonical_variant_name("bank") == "bank"
    config = ExperimentConfig(
        variant="memory_attention",
        memory_write_mode="dense",
        max_unique_tokens=1,
    )
    config.validate()
    model = build_variant("memory_attention", backbone(), memory_write_mode="dense")
    assert isinstance(model, BankVariant)
    assert model.variant_name == "memory_attention"

    multiscale = build_variant(
        "memory_attention_multiscale",
        MistralForCausalLM(
            micro_config(num_hidden_layers=3), attention_backend="reference"
        ),
        memory_dense_window=4,
        memory_sparse_window=3,
        memory_sparse_stride=8,
        memory_layers=[1, 2],
    )
    assert isinstance(multiscale, MultiscaleBankVariant)
    assert multiscale.variant_name == "memory_attention_multiscale"


def test_public_model_names_select_public_implementations():
    baseline = build_variant("swa_transformer", backbone())
    assert isinstance(baseline, SwaTransformerVariant)
    strided = build_variant(
        "strided_attention",
        MistralForCausalLM(micro_config(num_hidden_layers=1), attention_backend="reference"),
        sparse_attention_stride=8,
        sparse_attention_window=8,
    )
    assert isinstance(strided, StridedAttentionVariant)
    memory = build_variant(
        "strided_memory_attention",
        backbone(),
        memory_write_mode="strided",
        memory_write_stride=8,
    )
    assert isinstance(memory, MemoryAttentionVariant)
    multi = build_variant(
        "multiscale_memory_attention",
        MistralForCausalLM(micro_config(num_hidden_layers=2), attention_backend="reference"),
        memory_dense_window=2,
        memory_sparse_window=2,
        memory_sparse_stride=4,
    )
    assert isinstance(multi, MultiscaleMemoryAttentionVariant)
    hybrid = build_variant(
        "recirculation_strided_memory_attention",
        MistralForCausalLM(micro_config(num_hidden_layers=2), attention_backend="reference"),
        memory_write_mode="strided",
        memory_write_stride=8,
        recirculation_source_layer=1,
        recirculation_destination_layer=0,
    )
    assert isinstance(hybrid, RecirculationStridedMemoryAttentionVariant)


def test_factory_selects_only_requested_memory_layers_and_defaults_to_rope():
    multi_layer = MistralForCausalLM(
        micro_config(num_hidden_layers=3), attention_backend="reference"
    )
    model = build_variant(
        "bank",
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
            "bank",
            MistralForCausalLM(
                micro_config(num_hidden_layers=2), attention_backend="reference"
            ),
            memory_write_mode="periodic",
            memory_write_stride=8,
            memory_layers=[2],
        )


def test_factory_builds_multiscale_bank_without_a_write_policy_axis():
    model = build_variant(
        "bank_multiscale",
        MistralForCausalLM(
            micro_config(num_hidden_layers=3), attention_backend="reference"
        ),
        memory_dense_window=4,
        memory_sparse_window=3,
        memory_sparse_stride=8,
        memory_layers=[1, 2],
    )
    assert isinstance(model, MultiscaleBankVariant)
    assert model.memory_window == 7
    assert model.memory_layers == (1, 2)
    assert model.memory_write_mode == "dense"

    with pytest.raises(ValueError, match="does not accept memory_write"):
        build_variant(
            "bank_multiscale",
            backbone(),
            memory_write_mode="dense",
        )


@pytest.mark.parametrize(
    "legacy_name",
    ["memory_bank32", "dense_memory_bank", "sparse_memory_bank", "memory_add_sparse_bank"],
)
def test_factory_rejects_removed_bank_aliases(legacy_name):
    with pytest.raises(ValueError, match="unknown variant"):
        build_variant(legacy_name, backbone())


def test_factory_requires_memory_token_visibility_explicitly():
    with pytest.raises(ValueError, match="memory_token_visibility"):
        build_variant(
            "bank",
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


def test_factory_builds_adaptive_bank_recirculation_hybrid():
    model = build_variant(
        "bank_recirculation_hybrid",
        MistralForCausalLM(
            micro_config(num_hidden_layers=3), attention_backend="reference"
        ),
        memory_write_mode="periodic",
        memory_write_stride=8,
        memory_layers=[1],
        recirculation_source_layer=2,
        recirculation_destination_layer=0,
        recirculation_mode="adaptive",
    )
    assert isinstance(model, BankRecirculationHybridVariant)
    assert model.mode == "adaptive"
    assert model.memory_layers == (1,)
