from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from conftest import micro_config
from test_pass_depth_eval import make_artifact
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.data.packed_dataset import PackedTokenDataset, MemoryTokenPackedDataset
from tiny_mistral_mptt.evaluation import common
from tiny_mistral_mptt.evaluation.interventions import evaluate_memory_interventions
from tiny_mistral_mptt.evaluation.lm_eval_adapter import (
    generate_recurrent, score_token_continuation, score_token_continuation_recurrent,
)
from tiny_mistral_mptt.evaluation.nll import evaluate_nll
from tiny_mistral_mptt.evaluation.pass_depth import evaluate_pass_depth
from tiny_mistral_mptt.evaluation.recurrent import evaluate_recurrent_continuation
from tiny_mistral_mptt.evaluation.settings import resolve_evaluation_settings
from tiny_mistral_mptt.training.trainer import Trainer
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.variants.memory_attention import MemoryAttentionVariant
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.recurrent_memory import RecurrentMemoryVariant
from tiny_mistral_mptt.variants.vanilla import VanillaVariant


class Rows:
    """Unequal scored-token counts per physical block exercise token weighting."""
    manifest = SimpleNamespace(source_ids={"a": 0, "b": 1})
    split = "validation"

    def __init__(self, controls=False):
        self.rows = torch.tensor([
            [1, 7, 97 if controls else 3, 14, 97 if controls else 9, 22, 6, 12],
            [1, 8, 5, 16, 13, 25, 4, 11],
        ])
        self.sequence_length = self.rows.shape[1]

    def __len__(self):
        return len(self.rows)

    def batch(self, indices, *, device):
        return self.rows[indices].to(device)

    def source_id(self, index):
        return index


@pytest.mark.parametrize("merger", ["projected_residual", "recirculation"])
def test_hybrid_interventions_use_explicit_attention_channel_names(merger):
    backbone = MistralForCausalLM(micro_config(num_hidden_layers=2), attention_backend="reference")
    model = build_variant("dense_memory_attention", backbone, memory_layers=[1],
                          recurrent_merger=merger, recurrent_layers=[0])
    label = "recurrent"
    result = evaluate_memory_interventions(model, Rows(), device="cpu", max_blocks=1)
    expected = {
        "real_memory", "zero_memory", "mismatched_memory",
        f"zero_{label}_real_attention", f"mismatched_{label}_real_attention",
        f"real_{label}_zero_attention", f"real_{label}_mismatched_attention",
    }
    assert set(result["evaluation"]["policy"]["interventions"]) == expected
    assert all(result[name]["predicted_tokens"] == 7 for name in expected)


def make_model(kind="memory_add"):
    torch.manual_seed(48)
    backbone = MistralForCausalLM(micro_config(), attention_backend="reference")
    if kind == "vanilla":
        return VanillaVariant(backbone)
    if kind == "memory_token":
        return MemoryAttentionVariant(
            backbone, memory_write_mode="memory_token", memory_write_stride=2,
            memory_token_visibility="write_only", memory_window=3,
        )
    if kind in {"projected_residual", "recirculation"}:
        return RecurrentMemoryVariant(backbone, memory_layers=[1], merger=kind)
    model = MemoryAddVariant(backbone)
    with torch.no_grad():
        model.memory_projection.weight.copy_(0.05 * torch.eye(model.config.hidden_size))
    return model


@pytest.mark.parametrize("kind", ["vanilla", "memory_add", "memory_token", "projected_residual", "recirculation"])
def test_parallel_nll_pass_depth_and_manual_token_weighting_agree(kind):
    model = make_model(kind)
    data = Rows(controls=kind == "memory_token")
    passes = 1 if kind == "vanilla" else 3
    nll = evaluate_nll(model, data, device="cpu", passes=passes)
    depth = evaluate_pass_depth(model, data, device="cpu", passes=passes)
    assert model.training  # Evaluating does not leave a training model in eval mode.
    assert nll.nll == depth.final_nll
    assert nll.nll_by_source == depth.nll_by_source_by_pass[-1]
    assert nll.evaluation == depth.evaluation
    losses, counts = [], []
    with torch.no_grad():
        for index in range(len(data)):
            ids = data.batch([index], device="cpu")
            logits = (model(ids).logits if kind == "vanilla" else
                      model.compute_passes(ids, passes=passes).final.logits)
            labels = model.build_lm_labels(ids)
            count = int(labels.ne(-100).sum())
            loss = float(F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum"))
            losses.append(loss)
            counts.append(count)
    assert nll.predicted_tokens == sum(counts)
    assert nll.nll == sum(losses) / sum(counts)
    assert nll.predicted_tokens_by_source == dict(zip(("a", "b"), counts))
    assert nll.nll_by_source == dict(zip(("a", "b"), [v / c for v, c in zip(losses, counts)]))
    if kind == "memory_token":
        assert counts == [5, 7]


@pytest.mark.parametrize("precision", [None, "bfloat16"])
@pytest.mark.parametrize("passes", [1, 3])
def test_trainer_and_standalone_share_resolved_precision_and_results(tmp_path, monkeypatch, precision, passes):
    # Exercise BF16 dispatch on CPU without declaring it a supported production
    # mode or claiming that this qualifies any CUDA/MPS kernel.
    calls = []
    def cpu_test_autocast(device, dtype):
        calls.append((str(device), dtype))
        return torch.autocast("cpu", dtype=torch.bfloat16)
    monkeypatch.setattr(common, "autocast_context", cpu_test_autocast)
    root = tmp_path / "data"
    make_artifact(root)
    train, val = PackedTokenDataset(root, "train"), PackedTokenDataset(root, "validation")
    model = make_model()
    cfg = ExperimentConfig(
        variant="memory_add", device="cpu", model_dir="unused", data_dir=str(root),
        output_dir=str(tmp_path / "run"), eval_passes=passes, eval_batches=2,
        autocast_dtype=precision, attention_backend="reference",
    )
    trainer = Trainer(model=model, config=cfg, train_data=train, validation_data=val, device=torch.device("cpu"))
    record = trainer._evaluate()
    standalone = evaluate_nll(model, val, device="cpu", passes=passes, max_blocks=2, autocast_dtype=precision)
    assert record["nll"] == standalone.nll
    assert record["evaluation"] == standalone.evaluation
    assert record["predicted_tokens"] == standalone.predicted_tokens == 14
    assert record["weights"]["kind"] == "live_training_state"
    assert standalone.evaluation["precision"]["autocast_dtype"] == precision
    assert len(calls) == (2 if precision else 0)


def test_metadata_identifies_actual_prefix_and_memory_token_view(tmp_path):
    root = tmp_path / "data"
    make_artifact(root)
    data = MemoryTokenPackedDataset(PackedTokenDataset(root, "validation"), interval=2)
    result = evaluate_nll(make_model("memory_token"), data, device="cpu", max_blocks=1)
    identity = result.evaluation["data"]
    assert identity["selection"] == {"kind": "prefix_blocks", "start": 0, "stop": 1}
    assert identity["physical_sequence_length"] == 11
    assert identity["linguistic_sequence_length"] == 8
    assert identity["memory_token_interval"] == 2
    assert len(identity["manifest_sha256"]) == 64
    assert identity["declared_token_sha256"] == data.manifest.validation.data_sha256
    assert result.predicted_tokens == 7


def test_explicit_fp32_evaluation_disables_enclosing_autocast():
    model, data = make_model(), Rows()
    expected = evaluate_nll(model, data, device="cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = evaluate_nll(model, data, device="cpu", autocast_dtype=None)
        assert torch.is_autocast_enabled("cpu")  # outer scope restored
    assert actual == expected


def test_empty_target_selection_fails_and_restores_model_mode(monkeypatch):
    model = make_model().train()
    monkeypatch.setattr(model, "build_lm_labels", lambda ids: torch.full_like(ids, -100))
    with pytest.raises(ValueError, match="no linguistic prediction targets"):
        evaluate_nll(model, Rows(), device="cpu")
    assert model.training


@pytest.mark.parametrize("fields", [
    {"eval_prefill_passes": 0}, {"eval_decode_mode": "paper_recirculation"},
])
def test_invalid_experiment_evaluation_defaults_fail_early(fields):
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(fields)


@pytest.mark.parametrize("evaluator", [evaluate_nll, evaluate_pass_depth, evaluate_recurrent_continuation, evaluate_memory_interventions])
def test_all_packed_evaluators_restore_mode_when_forward_fails(monkeypatch, evaluator):
    model = make_model().train()
    def fail(*args, **kwargs):
        raise RuntimeError("injected forward failure")
    monkeypatch.setattr(model, "input_embeddings", fail)
    # The first pass goes through the backbone, while feedback goes through
    # input_embeddings. K=2 ensures the fault is reached in all four evaluators.
    kwargs = {"passes": 2} if evaluator in {evaluate_nll, evaluate_pass_depth} else {}
    if evaluator is evaluate_recurrent_continuation:
        kwargs = {"prefill_passes": 2, "prompt_tokens": 2, "continuation_tokens": 3}
    with pytest.raises(RuntimeError, match="injected forward failure"):
        evaluator(model, Rows(), device="cpu", **kwargs)
    assert model.training


@pytest.mark.parametrize("max_length,context,continuation", [
    (16, [1], [7, 8, 9]), (3, [1, 2, 3], [4, 5, 6]),
    (2, [1], [2, 3, 4, 5]), (1, [1, 2], [3, 4]), (2, [1], []),
])
def test_standard_cached_and_parallel_harness_score_identical_targets(max_length, context, continuation):
    model = make_model()
    parallel = score_token_continuation(model, device="cpu", max_length=max_length,
        context_enc=context, continuation_enc=continuation)
    cached = score_token_continuation_recurrent(model, device="cpu", max_length=max_length,
        context_enc=context, continuation_enc=continuation, prefill_passes=1, decode_mode="standard")
    assert cached[0] == pytest.approx(parallel[0], abs=2e-6)
    assert cached[1] == parallel[1]
    assert model.training


def test_diagnostic_and_intervention_targets_keep_existing_coverage():
    model, data = make_model("memory_token"), Rows(controls=True)
    result = evaluate_recurrent_continuation(model, data, device="cpu", prefill_passes=2,
        prompt_tokens=2, continuation_tokens=6, horizons=[6])
    assert result.predicted_tokens_per_mode == 10
    assert result.predicted_tokens_by_offset == (1, 2, 1, 2, 2, 2)
    assert result.horizons[0].predicted_tokens == 10
    assert result.predicted_tokens_by_source == {"a": 4, "b": 6}
    assert "vanilla_nll" not in asdict(result.horizons[0])
    assert "standard_k1_nll" in asdict(result.horizons[0])
    intervention = evaluate_memory_interventions(model, data, device="cpu")
    depth = evaluate_pass_depth(model, data, device="cpu", passes=2)
    assert intervention["baseline_pass1"]["nll"] == depth.nll_by_pass[0]
    assert intervention["real_memory"]["nll"] == depth.nll_by_pass[1]
    assert intervention["real_memory"]["predicted_tokens"] == 12


def test_experiment_defaults_and_independent_overrides_do_not_mutate_config():
    cfg = ExperimentConfig(variant="memory_add", eval_passes=4, autocast_dtype="bfloat16")
    before = cfg.to_dict()
    settings = resolve_evaluation_settings(cfg, make_model())
    assert (settings.passes, settings.prefill_passes, settings.decode_mode) == (4, 4, "feedback")
    override = resolve_evaluation_settings(cfg, make_model(), passes=2, prefill_passes=1,
        decode_mode="feedback", autocast_dtype="float32")
    assert (override.passes, override.prefill_passes, override.decode_mode) == (2, 1, "feedback")
    assert override.autocast_dtype is None
    assert cfg.to_dict() == before
    cfg.eval_prefill_passes, cfg.eval_decode_mode = 2, "standard"
    settings = resolve_evaluation_settings(cfg, make_model())
    assert settings.prefill_passes == 2 and settings.decode_mode == "standard"
    vanilla = resolve_evaluation_settings(ExperimentConfig(), make_model("vanilla"))
    assert vanilla.prefill_passes == 1 and vanilla.decode_mode == "standard"
    with pytest.raises(ValueError, match="does not implement feedback"):
        resolve_evaluation_settings(ExperimentConfig(), make_model("vanilla"), decode_mode="feedback")


@pytest.mark.parametrize("call", ["parallel", "feedback", "generation", "diagnostic", "intervention"])
def test_every_evaluation_lane_enters_requested_precision(monkeypatch, call):
    seen = []
    def cpu_test_autocast(device, dtype):
        seen.append(dtype)
        return torch.autocast("cpu", dtype=torch.bfloat16)
    monkeypatch.setattr(common, "autocast_context", cpu_test_autocast)
    model, data = make_model(), Rows()
    precision = {"autocast_dtype": "bfloat16"}
    if call == "parallel":
        score_token_continuation(model, device="cpu", max_length=8, context_enc=[1], continuation_enc=[7, 8], **precision)
    elif call == "feedback":
        score_token_continuation_recurrent(model, device="cpu", max_length=8, context_enc=[1],
            continuation_enc=[7, 8], prefill_passes=1, decode_mode="feedback", **precision)
    elif call == "generation":
        generate_recurrent(model, torch.tensor([[1]]), 2, prefill_passes=1, decode_mode="feedback", **precision)
    elif call == "diagnostic":
        evaluate_recurrent_continuation(model, data, device="cpu", prefill_passes=2,
            prompt_tokens=2, continuation_tokens=3, **precision)
    else:
        evaluate_memory_interventions(model, data, device="cpu", **precision)
    assert seen == ["bfloat16"]
    assert model.training
