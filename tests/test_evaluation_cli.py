import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

from test_evaluation_contract import make_model
from test_pass_depth_eval import make_artifact
from tiny_mistral_mptt.config import ExperimentConfig
from tiny_mistral_mptt.data.manifest import file_sha256
from tiny_mistral_mptt.training.checkpoint import save_checkpoint, TrainState


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script,extra", [
    ("evaluate_nll", []),
    ("evaluate_nll", ["--forward", "feedback"]),
    ("evaluate_pass_depth", []),
    ("evaluate_feedback_inference", ["--prompt-tokens", "2", "--continuation-tokens", "3"]),
    ("evaluate_memory_interventions", []),
])
def test_packed_cli_resolves_defaults_overrides_and_checkpoint_identity(tmp_path, monkeypatch, script, extra):
    root = tmp_path / "data"
    make_artifact(root)
    config = ExperimentConfig(
        variant="memory_add", model_dir="unused", data_dir=str(root),
        device="cpu", attention_backend="reference", autocast_dtype="bfloat16",
        eval_passes=4,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    checkpoint = tmp_path / "weights.pt"
    model = make_model()
    save_checkpoint(
        checkpoint, model=model, optimizer=torch.optim.AdamW(model.parameters()),
        sampler_state={}, train_state=TrainState(), experiment_config=config.to_dict(),
        data_manifest_sha256=file_sha256(root / "manifest.json"),
    )
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_variant_from_config", lambda *a, **k: make_model())
    output_path = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        script, "--config", str(config_path), "--checkpoint", str(checkpoint),
        "--device", "cpu", "--autocast-dtype", "float32", "--max-blocks", "1",
        "--output", str(output_path), *extra,
    ])
    module.main()
    doc = json.loads(output_path.read_text())
    expected_schema = {
        "evaluate_memory_interventions": 4,
        "evaluate_feedback_inference": 3,
    }.get(script, 2)
    assert doc["schema_version"] == expected_schema
    assert doc["provenance"]["normalized_config"]["autocast_dtype"] == "bfloat16"
    assert doc["provenance"]["weights"]["checkpoint_sha256"] == file_sha256(checkpoint)
    result = doc["results"][0] if "results" in doc else doc["result"]
    assert result["evaluation"]["precision"]["autocast_dtype"] is None
    assert result["evaluation"]["data"]["selection"]["stop"] == 1
    if script == "evaluate_nll" and extra:
        assert result["prefill_passes"] == 1
        assert result["predicted_tokens"] == 8 and result["aligned_predicted_tokens"] == 7
        assert doc["resolved_settings"]["forward_mode"] == "feedback"
    elif script in {"evaluate_nll", "evaluate_pass_depth"}:
        assert result["passes"] == 4  # no independent CLI default
        assert result["predicted_tokens"] == 7
    if script == "evaluate_feedback_inference":
        assert doc["prefill_passes"] == [4]
        assert result["predicted_tokens_per_mode"] == 3


def test_downstream_cli_defaults_and_actual_scoring_counts(tmp_path, monkeypatch):
    lm_eval = pytest.importorskip("lm_eval")
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {"[UNK]": 0, **{f"token{i}": i for i in range(1, 97)}}, unk_token="[UNK]",
    ))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    config = ExperimentConfig(
        variant="memory_add", model_dir=str(tmp_path), device="cpu",
        eval_passes=4, autocast_dtype="bfloat16",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    spec = importlib.util.spec_from_file_location("evaluate_lm_harness", ROOT / "scripts/evaluate_lm_harness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_variant_from_config", lambda *a, **k: make_model())
    def evaluate_locally(*, model, tasks, **kwargs):
        assert model.prefill_passes == 4 and model.decode_mode == "feedback"
        model._loglikelihood_tokens([(None, [1, 7], [8, 9])])
        return {"results": {tasks[0]: {"test": 1.0}}}
    monkeypatch.setattr(lm_eval, "simple_evaluate", evaluate_locally)
    output_path = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_lm_harness", "--config", str(config_path), "--initialized-baseline",
        "--suite", str(ROOT / "evaluation/suites/quick.yaml"),
        "--autocast-dtype", "float32", "--output", str(output_path),
    ])
    module.main()
    doc = json.loads(output_path.read_text())
    assert doc["resolved_settings"]["prefill_passes"] == 4
    assert doc["resolved_settings"]["decode_mode"] == "feedback"
    assert doc["precision"]["autocast_dtype"] is None
    assert doc["provenance"]["tokenizer"]["sha256"] == file_sha256(tmp_path / "tokenizer.json")
    assert doc["scoring"]["scored_tokens"] == 2 * len(doc["tasks"])
    assert all(task["scoring"]["scored_tokens"] == 2 for task in doc["tasks"].values())
