"""Removed execution policies must not silently become ordinary feedback."""

import importlib.util
from pathlib import Path

import pytest
import torch

from tiny_mistral_mptt import inference
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.evaluation.nll import evaluate_nll
from tiny_mistral_mptt.training.checkpoint import (
    TrainState,
    load_checkpoint,
    load_checkpoint_for_evaluation,
    load_model_weights,
    save_checkpoint,
)
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant


ROOT = Path(__file__).resolve().parents[1]
REMOVED_OPTIONS = [
    {"training_forward": "recirculation_bptt"},
    {"validation_forward": "paper_recirculation"},
    {"recirculation_activation_checkpointing": True},
    {"recirculation_bptt_truncate_tokens": 128},
]


@pytest.mark.parametrize("options", REMOVED_OPTIONS)
def test_removed_config_options_fail_explicitly(options):
    with pytest.raises(ValueError, match="removed"):
        ExperimentConfig.from_dict({**ExperimentConfig().to_dict(), **options})


def test_neutral_legacy_metadata_is_accepted_but_not_serialized():
    expected = ExperimentConfig().to_dict()
    loaded = ExperimentConfig.from_dict({
        **expected,
        "recirculation_activation_checkpointing": False,
        "recirculation_bptt_truncate_tokens": None,
    })
    assert loaded.to_dict() == expected


def test_paper_execution_is_not_exported_or_attached_to_the_merger_variant():
    for name in ("PaperRecirculationState", "prefill_paper_recirculation",
                 "paper_recirculation_decode_step"):
        assert not hasattr(inference, name)
    for name in ("compute_recirculation_logits", "compute_recirculation_bptt_loss",
                 "iter_recirculation_tbptt_losses", "_replay_upper_stack"):
        assert not hasattr(RecirculationVariant, name)
    assert hasattr(inference, "prefill_recurrent")
    assert hasattr(inference, "recurrent_decode_step")
    assert hasattr(inference, "prefill_exact")


def test_packed_nll_rejects_paper_execution_before_running_a_model():
    with pytest.raises(ValueError, match="removed"):
        evaluate_nll(None, [None], device="cpu", forward_mode="paper_recirculation")


def _checkpoint(tmp_path, options):
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = ExperimentConfig().to_dict()
    path = save_checkpoint(
        tmp_path / "state.pt", model=model, optimizer=optimizer,
        sampler_state={}, train_state=TrainState(unique_tokens_seen=8),
        experiment_config={**config, **options}, data_manifest_sha256="same-data",
    )
    return path, model, optimizer, config


def test_unaffected_multipass_era_checkpoint_still_resumes(tmp_path):
    path, model, optimizer, config = _checkpoint(tmp_path, {
        "recirculation_activation_checkpointing": False,
        "recirculation_bptt_truncate_tokens": None,
    })
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.zero_()
    state, _ = load_checkpoint(
        path, model=model, optimizer=optimizer,
        expected_experiment_config=config, expected_manifest_sha256="same-data",
    )
    assert state.unique_tokens_seen == 8
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    load_model_weights(path, model=model, expected_experiment_config=config)


@pytest.mark.parametrize("loader", ["resume", "evaluate", "weights"])
@pytest.mark.parametrize("options", REMOVED_OPTIONS)
def test_retired_policy_checkpoints_cannot_silently_load(tmp_path, loader, options):
    path, model, optimizer, config = _checkpoint(tmp_path, options)
    with pytest.raises(ValueError, match="removed"):
        if loader == "resume":
            load_checkpoint(
                path, model=model, optimizer=optimizer,
                expected_manifest_sha256="same-data",
            )
        elif loader == "evaluate":
            load_checkpoint_for_evaluation(path, model=model)
        else:
            load_model_weights(path, model=model, expected_experiment_config=config)


@pytest.mark.parametrize("script", ["benchmark_training_efficiency", "estimate_training_flops"])
def test_efficiency_tools_reject_retired_policy_without_running_it(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case = {"variant": "recirculation", "training_forward": "recirculation_bptt"}
    with pytest.raises(ValueError, match="removed"):
        if script == "benchmark_training_efficiency":
            module._run_case(case)
        else:
            module._estimate_case(None, case, {2: 1.0})
