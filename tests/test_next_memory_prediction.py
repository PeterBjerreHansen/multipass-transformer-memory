from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.nmp import (
    LatentPredictionHead,
    normalize_nmp_target,
    next_strict_true_indices,
    prepare_recurrent_nmp_alignment,
    prepare_bank_nmp_alignment,
    recurrent_nmp_pass_loss,
    bank_nmp_pass_loss,
)
from tiny_mistral_mptt.training.checkpoint import (
    TrainState,
    load_model_weights,
    save_checkpoint,
)
from tiny_mistral_mptt.training.phases import configure_phase


def backbone(seed: int = 1) -> MistralForCausalLM:
    torch.manual_seed(seed)
    return MistralForCausalLM(micro_config(), attention_backend="reference")


def memory_add_with_nmp(seed: int = 1):
    return build_variant(
        "memory_add",
        backbone(seed),
        recurrent_nmp_weight=0.1,
        nmp_projection_factor=1.3,
    )


def periodic_bank_with_nmp(seed: int = 1):
    return build_variant(
        "bank",
        backbone(seed),
        memory_write_mode="periodic",
        memory_write_stride=2,
        memory_layers=[0],
        bank_nmp_weight=0.1,
        nmp_projection_factor=1.3,
    )


def _make_head() -> LatentPredictionHead:
    return LatentPredictionHead(
        32,
        projection_factor=1.3,
        rms_norm_eps=1e-6,
        initialization_seed=99,
    )


def _activate_output(head: LatentPredictionHead) -> None:
    with torch.no_grad():
        head.output.weight.normal_(mean=0.0, std=0.02)
        head.output.bias.normal_(mean=0.0, std=0.02)


def test_predictor_api_accepts_only_current_hidden_state_and_starts_at_zero():
    head = _make_head()
    assert list(inspect.signature(head.forward).parameters) == ["hidden_states"]
    hidden = torch.randn(2, 5, 32)
    torch.testing.assert_close(head(hidden), torch.zeros_like(hidden), atol=0, rtol=0)


def test_predictor_construction_does_not_advance_global_rng():
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    _make_head()
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)


def test_rms_nmp_target_normalization_matches_model_rms_without_learned_gain():
    states = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    normalized = normalize_nmp_target(states, normalization="rms", eps=1e-6)
    expected = states * torch.rsqrt(states.square().mean(dim=-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(normalized, expected)
    torch.testing.assert_close(
        normalized.square().mean(dim=-1), torch.ones(1, 2), atol=1e-6, rtol=1e-6
    )


def test_nmp_target_normalization_rejects_unknown_mode():
    with pytest.raises(ValueError, match="target normalization"):
        normalize_nmp_target(
            torch.ones(1, 1, 2), normalization="norm_match", eps=1e-6
        )


def test_next_index_scan_is_strict_and_uses_sentinel_for_no_future_target():
    mask = torch.tensor([[False, True, False, True, True, False]])
    expected = torch.tensor([[1, 3, 3, 4, 6, 6]])
    torch.testing.assert_close(next_strict_true_indices(mask), expected)


def test_recurrent_nmp_skips_control_positions_and_targets_next_linguistic_token():
    ordinary = torch.tensor([[True, False, True, True, False]])
    target = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    # q0 -> target2, q2 -> target3. q3 has no future ordinary token.
    prediction = torch.tensor([[[2.0], [99.0], [3.0], [88.0], [77.0]]])
    loss, _, _ = recurrent_nmp_pass_loss(
        prediction, final_targets=target, ordinary_mask=ordinary
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=0, rtol=0)


def test_recurrent_target_is_detached_but_prediction_hidden_receives_gradient():
    prediction = torch.randn(1, 4, 3, requires_grad=True)
    target = torch.randn(1, 4, 3, requires_grad=True)
    ordinary = torch.ones(1, 4, dtype=torch.bool)
    loss, _, _ = recurrent_nmp_pass_loss(
        prediction, final_targets=target, ordinary_mask=ordinary
    )
    loss.backward()
    assert prediction.grad is not None and torch.count_nonzero(prediction.grad) > 0
    assert target.grad is None


def test_recurrent_nmp_empty_alignment_is_a_finite_differentiable_zero():
    prediction = torch.randn(1, 4, 3, requires_grad=True)
    target = torch.randn(1, 4, 3, requires_grad=True)
    ordinary = torch.tensor([[False, False, True, False]])
    loss, target_rms, target_std = recurrent_nmp_pass_loss(
        prediction, final_targets=target, ordinary_mask=ordinary
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert target_rms.item() == 0.0
    assert target_std.item() == 0.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert target.grad is None


def test_bank_nmp_uses_first_strictly_future_write_and_balances_write_events():
    # Writes at 2 and 5. Queries 0,1 map to event 2; queries 2,3,4 map
    # strictly forward to event 5. The first event has zero loss and the second
    # has SmoothL1(2,0)=1.5, so target-balanced loss is (0+1.5)/2=.75.
    ordinary = torch.ones(1, 6, dtype=torch.bool)
    writes = torch.tensor([[False, False, True, False, False, True]])
    target = torch.zeros(1, 6, 1)
    prediction = torch.tensor([[[0.0], [0.0], [2.0], [2.0], [2.0], [99.0]]])
    positions = torch.arange(6).view(1, 6)
    loss, _, _, distances = bank_nmp_pass_loss(
        prediction,
        final_written_states=target,
        ordinary_mask=ordinary,
        write_mask=writes,
        sequence_positions=positions,
    )
    torch.testing.assert_close(loss, torch.tensor(0.75))
    assert set(distances) == {"0", "1", "2_4", "5_8", "9_16", "17_32", "33_plus"}
    torch.testing.assert_close(distances["1"], torch.tensor(0.75))
    torch.testing.assert_close(distances["2_4"], torch.tensor(1.0))


def test_memory_token_write_can_have_zero_linguistic_distance_without_leakage():
    # Physical position 1 is a future control write, but it shares the
    # linguistic boundary of ordinary position 0.
    ordinary = torch.tensor([[True, False, True]])
    writes = torch.tensor([[False, True, False]])
    positions = torch.tensor([[0, 0, 1]])
    prediction = torch.zeros(1, 3, 1)
    targets = torch.zeros_like(prediction)
    _, _, _, distances = bank_nmp_pass_loss(
        prediction,
        final_written_states=targets,
        ordinary_mask=ordinary,
        write_mask=writes,
        sequence_positions=positions,
    )
    assert set(distances) == {"0", "1", "2_4", "5_8", "9_16", "17_32", "33_plus"}
    torch.testing.assert_close(distances["0"], torch.tensor(0.0))


def test_bank_nmp_with_no_future_write_is_a_finite_differentiable_zero():
    ordinary = torch.ones(1, 5, dtype=torch.bool)
    writes = torch.zeros(1, 5, dtype=torch.bool)
    positions = torch.arange(5).view(1, 5)
    prediction = torch.randn(1, 5, 2, requires_grad=True)
    target = torch.randn(1, 5, 2, requires_grad=True)
    loss, target_rms, target_std, distances = bank_nmp_pass_loss(
        prediction,
        final_written_states=target,
        ordinary_mask=ordinary,
        write_mask=writes,
        sequence_positions=positions,
    )
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert target_rms.item() == 0.0
    assert target_std.item() == 0.0
    assert all(value.item() == 0.0 for value in distances.values())
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert target.grad is None


def test_sparse_bank_model_with_no_future_write_keeps_ntp_finite():
    model = build_variant(
        "bank",
        backbone(),
        memory_write_mode="periodic",
        memory_write_stride=32,
        memory_layers=[0],
        bank_nmp_weight=0.1,
        nmp_projection_factor=1.3,
    )
    output = model.compute_loss(
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
        passes=2,
        loss_weights=[0.0, 1.0],
        nmp_weight_scale=1.0,
    )
    assert torch.isfinite(output.loss)
    assert output.metrics["bank_nmp_valid_queries"] == 0.0
    assert output.metrics["bank_nmp_loss"] == 0.0


@pytest.mark.parametrize("passes", [1, 2, 3])
def test_recurrent_predictions_at_t_are_invariant_to_tokens_after_t(passes: int):
    model = memory_add_with_nmp().eval()
    assert model.recurrent_nmp_predictor is not None
    _activate_output(model.recurrent_nmp_predictor)
    left = torch.tensor([[1, 2, 3, 4, 5, 6]])
    right = torch.tensor([[1, 2, 3, 70, 71, 72]])
    with torch.no_grad():
        left_runs = model._run_passes(left, passes=passes, phase="B")
        right_runs = model._run_passes(right, passes=passes, phase="B")
        left_prediction = model.recurrent_nmp_predictor(left_runs[-1].hidden_states)
        right_prediction = model.recurrent_nmp_predictor(right_runs[-1].hidden_states)
    torch.testing.assert_close(
        left_prediction[:, :3], right_prediction[:, :3], atol=1e-6, rtol=1e-6
    )


def test_bank_prediction_at_t_is_invariant_to_future_tokens():
    model = periodic_bank_with_nmp().eval()
    assert model.bank_nmp_predictor is not None
    _activate_output(model.bank_nmp_predictor)
    left = torch.tensor([[1, 2, 3, 4, 5, 6]])
    right = torch.tensor([[1, 2, 3, 70, 71, 72]])
    with torch.no_grad():
        left_run = model._run_passes(left, passes=2, phase="B")[-1]
        right_run = model._run_passes(right, passes=2, phase="B")[-1]
        left_prediction = model.bank_nmp_predictor(left_run.hidden_states)
        right_prediction = model.bank_nmp_predictor(right_run.hidden_states)
    torch.testing.assert_close(
        left_prediction[:, :3], right_prediction[:, :3], atol=1e-6, rtol=1e-6
    )


def test_bank_writer_gets_gradient_through_memory_using_hidden_not_target_branch():
    model = periodic_bank_with_nmp()
    assert model.bank_nmp_predictor is not None
    _activate_output(model.bank_nmp_predictor)
    # Bank readers are no-op initialized for safe backbone retrofitting. Move
    # the output projection off zero so this one-step test exercises the
    # causal writer -> reader -> h_t -> predictor path.
    with torch.no_grad():
        model.memory_readers["0"].o_proj.weight.normal_(mean=0.0, std=0.02)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = model.compute_loss(
        ids,
        phase="B",
        passes=2,
        loss_weights=[1.0, 0.0],
        nmp_weight_scale=1.0,
    )
    output.loss.backward()
    gradient = model.writer.proj.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_nmp_uses_uniform_pass_weights_independent_of_ntp_weights():
    model = memory_add_with_nmp()
    assert model.recurrent_nmp_predictor is not None
    _activate_output(model.recurrent_nmp_predictor)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = model.compute_loss(
        ids,
        passes=2,
        loss_weights=[0.0, 1.0],
        nmp_weight_scale=1.0,
    )
    expected = (
        output.metrics["recurrent_nmp_pass_1_loss"]
        + output.metrics["recurrent_nmp_pass_2_loss"]
    ) / 2.0
    assert output.metrics["recurrent_nmp_loss"] == pytest.approx(expected)

    weighted = model.compute_loss(
        ids,
        passes=2,
        loss_weights=[0.0, 1.0],
        recurrent_nmp_loss_weights=[1.0, 0.0],
        nmp_weight_scale=1.0,
    )
    assert weighted.metrics["recurrent_nmp_pass_1_weight"] == pytest.approx(1.0)
    assert weighted.metrics["recurrent_nmp_pass_2_weight"] == pytest.approx(0.0)
    assert weighted.metrics["recurrent_nmp_loss"] == pytest.approx(
        weighted.metrics["recurrent_nmp_pass_1_loss"]
    )


def test_bank_nmp_pass_weights_are_independent_and_uniform_by_default():
    model = periodic_bank_with_nmp().eval()
    assert model.bank_nmp_predictor is not None
    _activate_output(model.bank_nmp_predictor)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    default = model.compute_loss(ids, passes=2, loss_weights=[0.0, 1.0], nmp_weight_scale=1.0)
    assert default.metrics["bank_nmp_pass_1_weight"] == pytest.approx(0.5)
    assert default.metrics["bank_nmp_pass_2_weight"] == pytest.approx(0.5)
    weighted = model.compute_loss(
        ids,
        passes=2,
        loss_weights=[0.0, 1.0],
        bank_nmp_loss_weights=[0.0, 1.0],
        nmp_weight_scale=1.0,
    )
    assert weighted.metrics["bank_nmp_pass_1_weight"] == pytest.approx(0.0)
    assert weighted.metrics["bank_nmp_pass_2_weight"] == pytest.approx(1.0)
    assert weighted.metrics["bank_nmp_loss"] == pytest.approx(
        weighted.metrics["bank_nmp_pass_2_loss"]
    )


def test_every_pass_uses_one_shared_final_pass_recurrent_target(monkeypatch):
    import tiny_mistral_mptt.variants.multipass as multipass_module

    model = memory_add_with_nmp()
    seen: list[torch.Tensor] = []
    original = multipass_module.recurrent_nmp_pass_loss

    def recording_loss(predictions, *, alignment, diagnostics):
        seen.append(alignment.targets)
        return original(
            predictions,
            alignment=alignment,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(multipass_module, "recurrent_nmp_pass_loss", recording_loss)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = model.compute_loss(ids, passes=3, loss_weights=[0.0, 0.0, 1.0])
    assert len(seen) == 3
    assert all(target.data_ptr() == seen[0].data_ptr() for target in seen)
    assert not seen[0].requires_grad
    assert "recurrent_nmp_pass_3_loss" in output.metrics


def test_recirculation_nmp_uses_captured_internal_source_not_top_hidden():
    model = build_variant(
        "recirculation",
        backbone(),
        recirculation_source_layer=1,
        recirculation_destination_layer=0,
        recirculation_mode="adaptive",
        recurrent_nmp_weight=0.1,
    )
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    run = model._run_passes(ids, passes=2, phase="B")[-1]
    assert isinstance(run.feedback_source, torch.Tensor)
    selected = model._source_component(run, "recurrent")
    assert selected.data_ptr() == run.feedback_source.data_ptr()
    assert selected.data_ptr() != run.hidden_states.data_ptr()


def test_recurrent_nmp_defaults_to_detached_rms_normalized_source_target(monkeypatch):
    import tiny_mistral_mptt.variants.multipass as multipass_module

    model = memory_add_with_nmp().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    expected_run = model._run_passes(ids, passes=2, phase="B")[-1]
    expected = normalize_nmp_target(
        model._source_component(expected_run, "recurrent"),
        normalization="rms",
        eps=float(model.config.rms_norm_eps),
    )
    expected = prepare_recurrent_nmp_alignment(
        expected,
        ordinary_mask=torch.ones(1, 6, dtype=torch.bool),
    ).targets
    seen: list[torch.Tensor] = []
    original = multipass_module.recurrent_nmp_pass_loss

    def recording_loss(predictions, *, alignment, diagnostics):
        seen.append(alignment.targets)
        return original(
            predictions,
            alignment=alignment,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(multipass_module, "recurrent_nmp_pass_loss", recording_loss)
    model.compute_loss(ids, passes=2, loss_weights=[0.0, 1.0])
    assert len(seen) == 2
    torch.testing.assert_close(seen[0], expected)
    assert not seen[0].requires_grad
    valid = torch.ones(seen[0].shape[:2], dtype=torch.bool)
    torch.testing.assert_close(
        seen[0][valid].square().mean(), torch.ones(()), atol=1e-5, rtol=1e-5
    )


def test_recurrent_nmp_raw_target_remains_available_as_an_ablation():
    model = build_variant(
        "memory_add",
        backbone(),
        recurrent_nmp_weight=0.1,
        recurrent_nmp_target_normalization="none",
    ).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    run = model._run_passes(ids, passes=2, phase="B")[-1]
    source = model._source_component(run, "recurrent")
    assert model.recurrent_nmp_target_normalization == "none"
    torch.testing.assert_close(
        normalize_nmp_target(
            source, normalization=model.recurrent_nmp_target_normalization, eps=1e-6
        ),
        source,
    )


def test_recurrent_nmp_raw_ablation_still_detaches_target(monkeypatch):
    import tiny_mistral_mptt.variants.multipass as multipass_module

    model = build_variant(
        "memory_add",
        backbone(),
        recurrent_nmp_weight=0.1,
        recurrent_nmp_target_normalization="none",
    ).eval()
    seen: list[torch.Tensor] = []
    original = multipass_module.recurrent_nmp_pass_loss

    def recording_loss(predictions, *, alignment, diagnostics):
        seen.append(alignment.targets)
        return original(
            predictions,
            alignment=alignment,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(multipass_module, "recurrent_nmp_pass_loss", recording_loss)
    model.compute_loss(torch.tensor([[1, 2, 3, 4]]), passes=2)
    assert seen and not seen[0].requires_grad


def test_bank_nmp_target_is_final_pass_post_writer_state(monkeypatch):
    import tiny_mistral_mptt.variants.multipass as multipass_module

    model = periodic_bank_with_nmp()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    expected_runs = model._run_passes(ids, passes=2, phase="B")
    expected_written = model.writer(expected_runs[-1].hidden_states).detach()
    expected = prepare_bank_nmp_alignment(
        expected_written,
        ordinary_mask=torch.ones(1, 6, dtype=torch.bool),
        write_mask=model.write_mask(ids),
        sequence_positions=model.sequence_positions(ids),
    ).targets
    seen: list[torch.Tensor] = []
    original = multipass_module.bank_nmp_pass_loss

    def recording_loss(predictions, *, alignment, diagnostics):
        seen.append(alignment.targets)
        return original(
            predictions,
            alignment=alignment,
            diagnostics=diagnostics,
        )

    monkeypatch.setattr(multipass_module, "bank_nmp_pass_loss", recording_loss)
    model.compute_loss(ids, passes=2, loss_weights=[0.0, 1.0])
    assert len(seen) == 2
    assert all(target.data_ptr() == seen[0].data_ptr() for target in seen)
    for target in seen:
        torch.testing.assert_close(target, expected)
        assert not target.requires_grad


def test_dense_bank_next_write_is_exactly_next_physical_token():
    mask = torch.ones(2, 5, dtype=torch.bool)
    expected = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    torch.testing.assert_close(next_strict_true_indices(mask), expected)


def test_hybrid_heads_are_architecturally_equal_but_parameter_independent():
    model = build_variant(
        "bank_add_hybrid",
        backbone(),
        memory_write_mode="periodic",
        memory_write_stride=2,
        memory_layers=[0],
        recurrent_nmp_weight=0.1,
        bank_nmp_weight=0.2,
    )
    recurrent = model.recurrent_nmp_predictor
    bank = model.bank_nmp_predictor
    assert recurrent is not None and bank is not None
    assert type(recurrent) is type(bank)
    recurrent_shapes = [parameter.shape for parameter in recurrent.parameters()]
    bank_shapes = [parameter.shape for parameter in bank.parameters()]
    assert recurrent_shapes == bank_shapes
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(recurrent.parameters(), bank.parameters(), strict=True)
    )


def test_phase_a_trains_nmp_heads_but_freezes_pretrained_backbone():
    model = memory_add_with_nmp()
    configure_phase(model, "A")
    assert model.recurrent_nmp_predictor is not None
    added_ids = {id(parameter) for parameter in model.added_parameters()}
    assert all(
        id(parameter) in added_ids and parameter.requires_grad
        for parameter in model.recurrent_nmp_predictor.parameters()
    )
    assert not model.backbone.model.embed_tokens.weight.requires_grad


def test_enabling_training_only_head_does_not_change_forward_logits():
    plain = build_variant("memory_add", backbone(21)).eval()
    nmp = memory_add_with_nmp(seed=21).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        plain_logits = plain(ids, use_cache=False).logits
        nmp_logits = nmp(ids, use_cache=False).logits
    torch.testing.assert_close(nmp_logits, plain_logits, atol=0, rtol=0)


def test_zero_weight_factory_model_has_exact_historical_state_and_loss():
    historical = build_variant("memory_add", backbone(7))
    explicit_zero = build_variant(
        "memory_add",
        backbone(7),
        recurrent_nmp_weight=0.0,
        bank_nmp_weight=0.0,
    )
    assert historical.state_dict().keys() == explicit_zero.state_dict().keys()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    old = historical.compute_loss(ids, passes=2, loss_weights=[0.0, 1.0])
    new = explicit_zero.compute_loss(ids, passes=2, loss_weights=[0.0, 1.0])
    torch.testing.assert_close(old.loss, new.loss, atol=0, rtol=0)


def test_config_requires_supported_checkpointed_ramped_nmp_continuation():
    common = dict(
        variant="memory_add",
        pass_schedule=[{"probabilities": {2: 1.0}}],
        recurrent_nmp_weight=0.1,
    )
    with pytest.raises(ValueError, match="init_from or resume_from"):
        ExperimentConfig(**common).validate()
    with pytest.raises(ValueError, match="positive nmp_warmup_tokens"):
        ExperimentConfig(**common, init_from="ntp.pt").validate()
    valid = ExperimentConfig(
        **common, init_from="ntp.pt", nmp_warmup_tokens=1_000
    )
    valid.validate()
    assert valid.nmp_weight_scale_at(0) == 0.0
    assert valid.nmp_weight_scale_at(500) == 0.5
    assert valid.nmp_weight_scale_at(1_000) == 1.0


@pytest.mark.parametrize(
    ("variant", "recurrent", "bank"),
    [
        ("vanilla", 0.1, 0.0),
        ("fbt", 0.1, 0.0),
        ("memory_add", 0.0, 0.1),
        ("bank", 0.1, 0.0),
    ],
)
def test_config_rejects_objectives_without_architectural_targets(
    variant: str, recurrent: float, bank: float
):
    cfg = ExperimentConfig(
        variant=variant,
        recurrent_nmp_weight=recurrent,
        bank_nmp_weight=bank,
        nmp_warmup_tokens=10,
        init_from="ntp.pt",
    )
    with pytest.raises(ValueError, match="does not support"):
        cfg.validate()


def _save_model_checkpoint(path, model) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        sampler_state={},
        train_state=TrainState(),
        experiment_config={},
        data_manifest_sha256="test",
    )


def test_init_from_legacy_model_allows_only_whole_new_nmp_head(tmp_path):
    source = build_variant("memory_add", backbone(11))
    checkpoint = tmp_path / "ntp.pt"
    _save_model_checkpoint(checkpoint, source)
    target = memory_add_with_nmp(seed=11)
    provenance = load_model_weights(checkpoint, model=target)
    fresh = provenance["freshly_initialized_model_keys"]
    assert fresh
    assert all(name.startswith("recurrent_nmp_predictor.") for name in fresh)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, atol=0, rtol=0)


def test_init_from_rejects_partial_nmp_head(tmp_path):
    source = build_variant("memory_add", backbone(12))
    checkpoint = tmp_path / "partial.pt"
    _save_model_checkpoint(checkpoint, source)
    target = memory_add_with_nmp(seed=12)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    one_key = next(
        key for key in target.state_dict() if key.startswith("recurrent_nmp_predictor.")
    )
    payload["model"][one_key] = target.state_dict()[one_key]
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="partial"):
        load_model_weights(checkpoint, model=target)


def test_init_from_nmp_checkpoint_loads_every_key_without_fresh_state(tmp_path):
    source = memory_add_with_nmp(seed=15)
    checkpoint = tmp_path / "nmp.pt"
    _save_model_checkpoint(checkpoint, source)
    target = memory_add_with_nmp(seed=99)
    provenance = load_model_weights(checkpoint, model=target)
    assert provenance["freshly_initialized_model_keys"] == []
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, atol=0, rtol=0)


def test_nmp_checkpoint_roundtrip_is_strict():
    source = memory_add_with_nmp(seed=13)
    target = memory_add_with_nmp(seed=14)
    target.load_state_dict(source.state_dict(), strict=True)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, atol=0, rtol=0)
