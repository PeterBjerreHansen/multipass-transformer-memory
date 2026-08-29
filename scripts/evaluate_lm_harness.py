#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from numbers import Real
import os
from pathlib import Path
from typing import Any

import yaml

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.evaluation.lm_eval_adapter import make_lm_eval_adapter
from tiny_mistral_mptt.evaluation.provenance import (
    add_checkpoint_arguments,
    evaluation_provenance,
    load_evaluation_weights,
    render_or_write_json,
    seed_evaluation,
)
from tiny_mistral_mptt.model_factory import load_variant_from_config


def _scalar_response(value: Any) -> float | None:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, bool):
        return None
    if isinstance(value, Real):
        return float(value)
    return None


def _candidate_evidence(sample: dict[str, Any]) -> dict[str, Any] | None:
    raw = sample.get("filtered_resps")
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    scores = [_scalar_response(response) for response in raw]
    if any(score is None for score in scores):
        return None
    numeric_scores = [float(score) for score in scores if score is not None]
    ranked = sorted(
        range(len(numeric_scores)), key=numeric_scores.__getitem__, reverse=True
    )
    evidence: dict[str, Any] = {
        "candidate_loglikelihoods": numeric_scores,
        "predicted_index": ranked[0],
        "top_two_margin": numeric_scores[ranked[0]] - numeric_scores[ranked[1]],
    }
    gold = sample.get("target")
    if not isinstance(gold, int) or isinstance(gold, bool):
        doc = sample.get("doc")
        gold = doc.get("gold") if isinstance(doc, dict) else None
    if not isinstance(gold, int) or isinstance(gold, bool):
        doc = sample.get("doc")
        answer = doc.get("answer") if isinstance(doc, dict) else None
        if isinstance(answer, str) and answer.isdigit():
            one_based = int(answer)
            gold = one_based - 1 if 1 <= one_based <= len(numeric_scores) else None
    if (
        isinstance(gold, int)
        and not isinstance(gold, bool)
        and 0 <= gold < len(numeric_scores)
    ):
        best_other = max(
            score for index, score in enumerate(numeric_scores) if index != gold
        )
        evidence["gold_index"] = gold
        evidence["gold_margin"] = numeric_scores[gold] - best_other
    return evidence


def _annotate_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for sample in samples:
        copy = dict(sample)
        evidence = _candidate_evidence(copy)
        if evidence is not None:
            copy["candidate_evidence"] = evidence
        annotated.append(copy)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a checked-in, evidence-preserving lm-eval suite."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", default="evaluation/suites/quick.yaml")
    add_checkpoint_arguments(parser)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default=None,
        help="override the device declared by the experiment config",
    )
    parser.add_argument(
        "--decode-mode",
        choices=("standard", "feedback"),
        required=True,
        help="continuation mechanism; independent of prompt prefill depth K",
    )
    parser.add_argument(
        "--prefill-passes",
        type=int,
        required=True,
        help="number of full prompt passes; this does not select decode mode",
    )
    parser.add_argument("--max-gen-tokens", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--numpy-random-seed", type=int, default=1234)
    parser.add_argument("--torch-random-seed", type=int, default=1234)
    parser.add_argument("--fewshot-random-seed", type=int, default=1234)
    parser.add_argument(
        "--confirm-run-unsafe-code",
        action="store_true",
        help="required by code-generation tasks that execute generated programs",
    )
    args = parser.parse_args()
    if args.prefill_passes < 1:
        raise SystemExit("--prefill-passes must be positive")

    try:
        import lm_eval
    except ImportError as exc:
        raise SystemExit(
            "install evaluation dependencies with: uv sync --extra eval"
        ) from exc

    seeds = {
        "random": args.random_seed,
        "numpy": args.numpy_random_seed,
        "torch": args.torch_random_seed,
        "fewshot": args.fewshot_random_seed,
    }
    seed_evaluation(args.torch_random_seed)
    cfg = load_experiment_config(args.config)
    device = resolve_device(cfg.device if args.device is None else args.device)
    model = load_variant_from_config(cfg, device=device)
    weights = load_evaluation_weights(
        model=model,
        config=cfg,
        checkpoint=args.checkpoint,
        initialized_baseline=args.initialized_baseline,
    )
    adapter = make_lm_eval_adapter(
        model,
        tokenizer_path=Path(cfg.model_dir) / "tokenizer.json",
        device=device,
        max_gen_toks=args.max_gen_tokens,
        decode_mode=args.decode_mode,
        prefill_passes=args.prefill_passes,
    )

    suite_path = Path(args.suite)
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    suite_kind = suite.get("kind")
    if suite_kind not in {"candidate_loglikelihood", "generation"}:
        raise SystemExit(
            "suite must declare kind: candidate_loglikelihood or generation"
        )

    tasks: dict[str, Any] = {}
    for task in suite["tasks"]:
        name = task["name"]
        result = lm_eval.simple_evaluate(
            model=adapter,
            tasks=[name],
            num_fewshot=task.get("num_fewshot"),
            limit=args.limit,
            log_samples=True,
            random_seed=args.random_seed,
            numpy_random_seed=args.numpy_random_seed,
            torch_random_seed=args.torch_random_seed,
            fewshot_random_seed=args.fewshot_random_seed,
            confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        )
        task_samples = {
            task_name: _annotate_samples(samples)
            for task_name, samples in result.get("samples", {}).items()
        }
        tasks[name] = {
            "metrics": result["results"],
            "samples": task_samples,
            "task_configs": result.get("configs", {}),
            "task_versions": result.get("versions", {}),
            "num_fewshot": result.get("n-shot", {}),
            "higher_is_better": result.get("higher_is_better", {}),
        }
        print(
            name,
            json.dumps(tasks[name]["metrics"], sort_keys=True, default=str),
        )

    document = {
        "evaluation_kind": suite_kind,
        "semantics": {
            "prefill_passes": args.prefill_passes,
            "decode_mode": args.decode_mode,
            "candidate_scoring": (
                "teacher_forced_observed_tokens_with_live_feedback"
                if suite_kind == "candidate_loglikelihood"
                and args.decode_mode == "feedback"
                else None
            ),
        },
        "limit": args.limit,
        "provenance": evaluation_provenance(
            config_path=args.config,
            config=cfg,
            weight_identity=weights,
            device=device,
            seeds=seeds,
            suite_path=suite_path,
        ),
        "tasks": tasks,
    }
    rendered = render_or_write_json(document, args.output)
    if not args.output:
        print(rendered)


if __name__ == "__main__":
    main()
