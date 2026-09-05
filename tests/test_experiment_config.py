import pytest

from tiny_mistral_mptt.config import ExperimentConfig, load_experiment_config


def _config(**overrides):
    values = {
        "variant": "fbt",
        "model_dir": "unused",
        "data_dir": "unused",
        "output_dir": "unused",
        "pass_schedule": [{"probabilities": {2: 0.5, 3: 0.5}}],
        "eval_every_tokens": 0,
        "checkpoint_every_tokens": 0,
    }
    values.update(overrides)
    cfg = ExperimentConfig(**values)
    cfg.validate()
    return cfg


def test_k_specific_pass_weights_are_selected_exactly():
    cfg = _config(
        pass_loss_weights_by_k={
            2: [0.25, 0.75],
            3: [0.05, 0.20, 0.75],
        }
    )

    assert cfg.loss_weights_for_passes(2) == [0.25, 0.75]
    assert cfg.loss_weights_for_passes(3) == [0.05, 0.20, 0.75]


def test_k_specific_weights_require_exact_schedule_coverage():
    with pytest.raises(ValueError, match="exactly match"):
        _config(pass_loss_weights_by_k={2: [0.25, 0.75]})

    with pytest.raises(ValueError, match="exactly match"):
        _config(pass_loss_weights_by_k={2: [0.25, 0.75], 3: [0.05, 0.20, 0.75], 4: [1.0]})


def test_global_and_k_specific_weights_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _config(
            pass_loss_weights=[0.25, 0.75],
            pass_loss_weights_by_k={
                2: [0.25, 0.75],
                3: [0.05, 0.20, 0.75],
            },
        )


def test_yaml_style_string_k_keys_are_canonicalized():
    cfg = _config(
        pass_loss_weights_by_k={
            "2": [0.25, 0.75],
            "3": [0.05, 0.20, 0.75],
        }
    )

    assert cfg.pass_loss_weights_by_k == {
        2: [0.25, 0.75],
        3: [0.05, 0.20, 0.75],
    }


def test_ntp_pass_weights_are_canonical():
    cfg = _config(
        ntp_pass_loss_weights_by_k={2: [0.25, 0.75], 3: [0.1, 0.2, 0.7]},
    )

    assert cfg.ntp_loss_weights_for_passes(2) == [0.25, 0.75]
    serialized = cfg.to_dict()
    assert serialized["ntp_pass_loss_weights_by_k"] == {
        2: [0.25, 0.75],
        3: [0.1, 0.2, 0.7],
    }
    assert "pass_loss_weights" not in serialized
    assert "pass_loss_weights_by_k" not in serialized

def test_bfloat16_autocast_requires_fp32_parameter_storage():
    cfg = _config(dtype="float32", autocast_dtype="bfloat16")
    assert cfg.autocast_dtype == "bfloat16"

    with pytest.raises(ValueError, match="requires dtype=float32"):
        _config(dtype="bfloat16", autocast_dtype="bfloat16")


def test_unvalidated_autocast_dtype_is_rejected():
    with pytest.raises(ValueError, match="autocast_dtype"):
        _config(autocast_dtype="float16")

def test_memory_config_requires_coherent_write_policy():
    dense = _config(variant="memory_attention", memory_write_mode="dense")
    assert dense.memory_write_mode == "dense"
    assert dense.memory_write_stride is None
    assert dense.memory_token_visibility is None

    periodic = _config(
        variant="memory_attention",
        memory_write_mode="periodic",
        memory_write_stride=4,
        memory_layers=[7, 3],
    )
    assert periodic.memory_write_stride == 4
    assert periodic.memory_layers == [3, 7]
    assert periodic.memory_position_encoding == "rope"
    strided = _config(
        variant="memory_attention",
        memory_write_mode="strided",
        memory_write_stride=4,
    )
    assert strided.memory_write_mode == "strided"

    mem = _config(
        variant="memory_token_attention",
        recurrent_merger="projected_residual", recurrent_layers=[3],
        memory_write_mode="memory_token",
        memory_write_stride=8,
        memory_token_visibility="visible",
    )
    assert mem.memory_write_mode == "memory_token"
    assert mem.memory_token_visibility == "visible"

    with pytest.raises(ValueError, match="require memory_write_mode"):
        _config(variant="memory_attention")
    with pytest.raises(ValueError, match="requires positive memory_write_stride"):
        _config(variant="memory_attention", memory_write_mode="periodic")
    with pytest.raises(ValueError, match="requires memory_token_visibility"):
        _config(
            variant="memory_attention",
            memory_write_mode="memory_token",
            memory_write_stride=8,
        )
    with pytest.raises(ValueError, match="must not set memory_write_stride"):
        _config(variant="memory_attention", memory_write_mode="dense", memory_write_stride=1)
    with pytest.raises(ValueError, match="applies only"):
        _config(
            variant="memory_attention",
            memory_write_mode="periodic",
            memory_write_stride=8,
            memory_token_visibility="write_only",
        )


def test_memory_fields_cannot_silently_change_other_variants():
    with pytest.raises(ValueError, match="supported only for Memory Attention variants"):
        _config(variant="memory_add", memory_write_stride=4)
    with pytest.raises(ValueError, match="supported only for Memory Attention variants"):
        _config(variant="memory_add", memory_layers=[3])


def test_dense_and_strided_memory_config_uses_explicit_dense_and_sparse_windows():
    cfg = _config(
        variant="dense_and_strided_memory_attention",
        memory_dense_window=32,
        memory_sparse_window=32,
        memory_sparse_stride=32,
        memory_layers=[7, 4],
    )
    assert cfg.memory_layers == [4, 7]
    assert cfg.memory_position_encoding == "rope"
    assert cfg.memory_window == 64

    with pytest.raises(ValueError, match="requires memory_dense_window"):
        _config(variant="dense_and_strided_memory_attention")
    with pytest.raises(ValueError, match="at least one non-zero"):
        _config(
            variant="dense_and_strided_memory_attention",
            memory_dense_window=0,
            memory_sparse_window=0,
            memory_sparse_stride=32,
        )
    with pytest.raises(ValueError, match="conflicts with memory_write_mode"):
        _config(
            variant="dense_and_strided_memory_attention",
            memory_write_mode="dense",
            memory_dense_window=32,
            memory_sparse_window=32,
            memory_sparse_stride=32,
        )


def test_strided_self_attention_config_is_single_pass_and_rejects_memory_fields():
    cfg = _config(
        variant="strided_self_attention",
        pass_schedule=[{"probabilities": {1: 1.0}}],
        sparse_attention_stride=32,
        sparse_attention_window=32,
        sparse_attention_layers=[7, 3],
    )
    assert cfg.sparse_attention_layers == [3, 7]

    with pytest.raises(ValueError, match="requires positive sparse_attention_stride"):
        _config(
            variant="strided_self_attention",
            pass_schedule=[{"probabilities": {1: 1.0}}],
            sparse_attention_window=32,
        )
    with pytest.raises(ValueError, match="only one-pass"):
        _config(
            variant="strided_self_attention",
            sparse_attention_stride=32,
            sparse_attention_window=32,
        )
    with pytest.raises(ValueError, match="supported only for Memory Attention variants"):
        _config(
            variant="strided_self_attention",
            pass_schedule=[{"probabilities": {1: 1.0}}],
            sparse_attention_stride=32,
            sparse_attention_window=32,
            memory_layers=[3],
        )


def test_memory_attention_layer_and_position_configuration_is_validated():
    cfg = _config(
        variant="memory_attention",
        memory_write_mode="periodic",
        memory_write_stride=32,
        memory_layers="all",
        memory_position_encoding="none",
    )
    assert cfg.memory_layers == "all"
    assert cfg.memory_position_encoding == "none"

    with pytest.raises(ValueError, match="non-empty list"):
        _config(
            variant="memory_attention",
            memory_write_mode="periodic",
            memory_write_stride=32,
            memory_layers=[],
        )
    with pytest.raises(ValueError, match="unique"):
        _config(
            variant="memory_attention",
            memory_write_mode="periodic",
            memory_write_stride=32,
            memory_layers=[3, 3],
        )
    with pytest.raises(ValueError, match=r"rope\|none"):
        _config(
            variant="memory_attention",
            memory_write_mode="periodic",
            memory_write_stride=32,
            memory_position_encoding="absolute",
        )


def test_middle_layer_recirculation_config_is_archived():
    with pytest.raises(ValueError, match="variant must be one of"):
        _config(
            variant="recirculation",
            recirculation_source_layer=3,
            recirculation_destination_layer=1,
        )


def test_recirculation_inspired_controller_width_is_explicit():
    cfg = _config(
        variant="recurrent_memory",
        phase="A",
        memory_window=1,
        memory_layers=[3],
        recurrent_merger="recirculation",
        recurrent_controller_hidden_size=660,
    )
    assert cfg.recurrent_controller_hidden_size == 660
    with pytest.raises(ValueError, match="requires recurrent_merger=recirculation"):
        _config(
            variant="recurrent_memory",
            phase="A",
            memory_window=1,
            memory_layers=[3],
            recurrent_merger="projected_residual",
            recurrent_controller_hidden_size=660,
        )


def test_integrated_freeze_policy_requires_a_later_phase_b_transition() -> None:
    cfg = _config(freeze_pretrained_until_tokens=50, max_unique_tokens=100)
    assert cfg.freeze_pretrained_until_tokens == 50

    with pytest.raises(ValueError, match="integrated Phase-B"):
        _config(
            phase="A",
            freeze_pretrained_until_tokens=50,
            max_unique_tokens=100,
        )
    with pytest.raises(ValueError, match="below max_unique_tokens"):
        _config(freeze_pretrained_until_tokens=100, max_unique_tokens=100)


def test_optional_recurrent_memory_requires_its_own_read_layers():
    cfg = _config(
        variant="memory_attention",
        phase="A",
        memory_write_mode="periodic",
        memory_write_stride=32,
        memory_layers=[3, 7],
        recurrent_merger="recirculation",
        recurrent_layers=[5, 3],
    )
    assert cfg.memory_position_encoding == "rope"
    assert cfg.recurrent_merger == "recirculation"
    assert cfg.recurrent_layers == [3, 5]

    with pytest.raises(ValueError, match="recurrent_layers"):
        _config(
            variant="memory_attention",
            memory_write_mode="periodic",
            memory_write_stride=32,
            recurrent_merger="recirculation",
        )


def test_recirculation_fields_cannot_silently_change_other_variants():
    with pytest.raises(ValueError, match="apply only to recirculation"):
        _config(
            variant="memory_add",
            recirculation_source_layer=3,
            recirculation_destination_layer=1,
        )


def test_fbt_paper_recipe_fields_are_explicit_and_variant_scoped():
    cfg = _config(
        fbt_normalize_gate_input=True,
        fbt_latent_jitter_std=0.02,
        prefix_mixin_probability=1.0,
    )
    assert cfg.fbt_normalize_gate_input is True
    assert cfg.fbt_latent_jitter_std == 0.02

    with pytest.raises(ValueError, match=r"fbt_\* fields"):
        _config(variant="memory_add", fbt_normalize_gate_input=True)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _config(fbt_latent_jitter_std=-0.01)


def test_early_stop_pass_depth_gates_are_canonicalized_and_validated():
    cfg = _config(
        eval_every_tokens=8,
        eval_passes=4,
        early_stop={
            "pass_nll_max": {"1": 2.573, "4": 2.33},
            "pass_nll_delta_max": [
                {"pass": 4, "reference_pass": 2, "max_delta": 0.02}
            ],
            "hidden_delta_nonincreasing": True,
        },
    )
    assert cfg.early_stop == {
        "pass_nll_max": {1: 2.573, 4: 2.33},
        "pass_nll_delta_max": [
            {"pass": 4, "reference_pass": 2, "max_delta": 0.02}
        ],
        "hidden_delta_nonincreasing": True,
    }

    with pytest.raises(ValueError, match="positive eval_every_tokens"):
        _config(
            eval_passes=4,
            early_stop={"pass_nll_max": {4: 2.33}},
        )
    with pytest.raises(ValueError, match="beyond eval_passes"):
        _config(
            eval_every_tokens=8,
            eval_passes=3,
            early_stop={"pass_nll_max": {4: 2.33}},
        )


def test_relative_config_inheritance_overrides_only_child_fields(tmp_path):
    base = tmp_path / "base.yml"
    child_dir = tmp_path / "rates"
    child_dir.mkdir()
    child = child_dir / "arm.yaml"
    base.write_text(
        "\n".join(
            (
                "variant: fbt",
                "model_dir: shared-model",
                "data_dir: shared-data",
                "output_dir: base-output",
                "learning_rate: 0.0001",
            )
        ),
        encoding="utf-8",
    )
    child.write_text(
        "extends: ../base.yml\noutput_dir: child-output\nlearning_rate: 0.0003\n",
        encoding="utf-8",
    )

    cfg = load_experiment_config(child)

    assert cfg.variant == "fbt"
    assert cfg.model_dir == "shared-model"
    assert cfg.data_dir == "shared-data"
    assert cfg.output_dir == "child-output"
    assert cfg.learning_rate == pytest.approx(3e-4)


def test_config_inheritance_rejects_cycles_and_absolute_parents(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic experiment config inheritance"):
        load_experiment_config(first)

    absolute = tmp_path / "absolute.yaml"
    absolute.write_text(f"extends: {first}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extends must be relative"):
        load_experiment_config(absolute)
