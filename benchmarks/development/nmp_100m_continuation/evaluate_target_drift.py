#!/usr/bin/env python
"""Compare fixed-block NMP targets between the 100M parent and a continuation."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.nmp import normalize_nmp_target, prepare_recurrent_nmp_alignment
from tiny_mistral_mptt.training.checkpoint import (
    load_checkpoint_for_evaluation,
    load_model_weights,
)


def _root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("repository root not found")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _root() / path


@torch.no_grad()
def _target_vectors(model, dataset, *, device, passes, max_blocks, max_vectors):
    model.eval()
    collected: list[torch.Tensor] = []
    remaining = max_vectors
    for index in range(min(len(dataset), max_blocks)):
        ids = dataset.batch([index], device=device)
        runs = model._run_passes(ids, passes=passes, phase="B")
        target_runs = runs if model.nmp_target_mode == "same_pass" else (runs[-1],)
        ordinary_mask = ~model.control_token_mask(ids)
        for run in target_runs:
            vectors: list[torch.Tensor] = []
            if model.recurrent_nmp_predictor is not None:
                target = normalize_nmp_target(
                    model._source_component(run, "recurrent"),
                    normalization=model.recurrent_nmp_target_normalization,
                    eps=float(model.config.rms_norm_eps),
                )
                alignment = prepare_recurrent_nmp_alignment(
                    target, ordinary_mask=ordinary_mask
                )
                vectors.append(alignment.targets[alignment.valid])
            if model.bank_nmp_predictor is not None:
                written = model.nmp_written_states(
                    model._source_component(run, "bank")
                )
                write_mask = model.nmp_write_mask(ids)
                from tiny_mistral_mptt.nmp import prepare_bank_nmp_alignment

                alignment = prepare_bank_nmp_alignment(
                    written,
                    ordinary_mask=ordinary_mask,
                    write_mask=write_mask,
                    sequence_positions=model.nmp_sequence_positions(ids),
                )
                # One vector per target event matches the event-balanced
                # Memory-Attention objective rather than repeating per query.
                vectors.append(written[alignment.present_events])
            for values in vectors:
                if remaining <= 0:
                    break
                values = values[:remaining].detach().float().cpu()
                collected.append(values)
                remaining -= values.shape[0]
        if remaining <= 0:
            break
    if not collected:
        raise RuntimeError("no valid NMP target vectors were found")
    return torch.cat(collected, dim=0)


def _spectrum(values: torch.Tensor) -> dict[str, float]:
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if float(total) == 0.0:
        return {"effective_rank": 0.0, "top_1_fraction": 0.0, "top_10_fraction": 0.0}
    probabilities = eigenvalues / total
    positive = probabilities[probabilities > 0]
    effective_rank = torch.exp(-(positive * positive.log()).sum())
    descending = eigenvalues.flip(0)
    return {
        "effective_rank": float(effective_rank),
        "top_1_fraction": float(descending[:1].sum() / total),
        "top_10_fraction": float(descending[:10].sum() / total),
    }


def _linear_cka(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    x = reference - reference.mean(dim=0, keepdim=True)
    y = candidate - candidate.mean(dim=0, keepdim=True)
    cross = (x.T @ y).square().sum()
    x_norm = (x.T @ x).square().sum().sqrt()
    y_norm = (y.T @ y).square().sum().sqrt()
    denominator = x_norm * y_norm
    return float(cross / denominator) if float(denominator) > 0 else 0.0


def _release(device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--max-vectors", type=int, default=4096)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.passes < 1 or args.max_blocks < 1 or args.max_vectors < 2:
        raise SystemExit("passes/blocks must be positive and max-vectors must be at least two")

    config_path = _resolve(args.config)
    cfg = load_experiment_config(config_path)
    device = resolve_device(cfg.device)
    dataset = load_packed_dataset_for_experiment(
        cfg.data_dir,
        "validation",
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
    )
    reference_path = _resolve(args.reference_checkpoint)
    candidate_path = _resolve(args.candidate_checkpoint)

    reference_model = load_variant_from_config(cfg, device=device)
    load_model_weights(
        reference_path,
        model=reference_model,
        expected_experiment_config=cfg.to_dict(),
    )
    reference = _target_vectors(
        reference_model,
        dataset,
        device=device,
        passes=args.passes,
        max_blocks=args.max_blocks,
        max_vectors=args.max_vectors,
    )
    del reference_model
    _release(device)

    candidate_model = load_variant_from_config(cfg, device=device)
    load_checkpoint_for_evaluation(
        candidate_path,
        model=candidate_model,
        expected_experiment_config=cfg.to_dict(),
    )
    candidate = _target_vectors(
        candidate_model,
        dataset,
        device=device,
        passes=args.passes,
        max_blocks=args.max_blocks,
        max_vectors=args.max_vectors,
    )
    del candidate_model
    _release(device)

    count = min(reference.shape[0], candidate.shape[0])
    reference = reference[:count]
    candidate = candidate[:count]
    cosine = F.cosine_similarity(reference, candidate, dim=-1)
    difference = candidate - reference
    report = {
        "config": str(config_path),
        "reference_checkpoint": str(reference_path),
        "candidate_checkpoint": str(candidate_path),
        "passes": args.passes,
        "target_mode": cfg.nmp_target_mode,
        "vectors": count,
        "mean_paired_cosine": float(cosine.mean()),
        "linear_cka": _linear_cka(reference, candidate),
        "reference_rms": float(reference.square().mean().sqrt()),
        "candidate_rms": float(candidate.square().mean().sqrt()),
        "difference_rms": float(difference.square().mean().sqrt()),
        "reference_spectrum": _spectrum(reference),
        "candidate_spectrum": _spectrum(candidate),
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
