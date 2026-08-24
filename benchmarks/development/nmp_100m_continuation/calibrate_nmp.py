#!/usr/bin/env python
"""Measure NMP gradient pressure after an isolated predictor-head warm-up."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import torch

from tiny_mistral.device import resolve_device
from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.data.packed_dataset import load_packed_dataset_for_experiment
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.precision import autocast_context
from tiny_mistral_mptt.training.checkpoint import load_model_weights


def _root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("repository root not found")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _root() / path


def _predictor_parameters(model) -> list[torch.nn.Parameter]:
    result: list[torch.nn.Parameter] = []
    for name in ("recurrent_nmp_predictor", "bank_nmp_predictor"):
        predictor = getattr(model, name, None)
        if predictor is not None:
            result.extend(predictor.parameters())
    return result


def _parameter_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    heads = _predictor_parameters(model)
    head_ids = {id(parameter) for parameter in heads}
    added_ids = {id(parameter) for parameter in model.added_parameters()}
    pretrained = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in added_ids
    ]
    memory = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
        and id(parameter) in added_ids
        and id(parameter) not in head_ids
    ]
    return {"pretrained": pretrained, "memory": memory, "head": heads}


def _gradients(
    loss: torch.Tensor,
    groups: dict[str, Sequence[torch.nn.Parameter]],
) -> dict[str, list[torch.Tensor]]:
    names = list(groups)
    parameters = [parameter for name in names for parameter in groups[name]]
    values = torch.autograd.grad(loss, parameters, allow_unused=True)
    result: dict[str, list[torch.Tensor]] = {}
    offset = 0
    for name in names:
        count = len(groups[name])
        result[name] = [
            torch.zeros_like(parameter) if value is None else value.detach()
            for parameter, value in zip(
                groups[name], values[offset : offset + count], strict=True
            )
        ]
        offset += count
    return result


def _subtract(
    left: dict[str, list[torch.Tensor]],
    right: dict[str, list[torch.Tensor]],
) -> dict[str, list[torch.Tensor]]:
    return {
        name: [
            a - b for a, b in zip(left[name], right[name], strict=True)
        ]
        for name in left
    }


def _norm(values: Sequence[torch.Tensor]) -> float:
    if not values:
        return 0.0
    total = torch.zeros((), dtype=torch.float32, device=values[0].device)
    for value in values:
        total = total + value.float().square().sum()
    return float(total.sqrt().cpu())


def _cosine(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float | None:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    dot = sum(
        (a.float() * b.float()).sum()
        for a, b in zip(left, right, strict=True)
    )
    return float((dot / (left_norm * right_norm)).cpu())


def _objective(model, ids, cfg, *, recurrent: float, bank: float):
    old_recurrent = model.recurrent_nmp_weight
    old_bank = model.bank_nmp_weight
    model.recurrent_nmp_weight = float(recurrent)
    model.bank_nmp_weight = float(bank)
    try:
        return model.compute_loss(
            ids,
            phase=cfg.phase,
            passes=2,
            loss_weights=cfg.ntp_loss_weights_for_passes(2),
            recurrent_nmp_loss_weights=cfg.recurrent_nmp_loss_weights_for_passes(2),
            bank_nmp_loss_weights=cfg.bank_nmp_loss_weights_for_passes(2),
            nmp_weight_scale=1.0,
        )
    finally:
        model.recurrent_nmp_weight = old_recurrent
        model.bank_nmp_weight = old_bank


def _measure(model, ids, cfg, groups) -> dict[str, float | None]:
    ntp_output = _objective(model, ids, cfg, recurrent=0.0, bank=0.0)
    ntp = _gradients(ntp_output.loss, groups)
    result: dict[str, float | None] = {
        "ntp_loss": float(ntp_output.metrics["ntp_loss"]),
    }
    for group, gradients in ntp.items():
        result[f"ntp_{group}_gradient_norm"] = _norm(gradients)

    objectives = []
    if model.recurrent_nmp_predictor is not None:
        objectives.append(("recurrent", 1.0, 0.0))
    if model.bank_nmp_predictor is not None:
        objectives.append(("bank", 0.0, 1.0))
    for name, recurrent, bank in objectives:
        output = _objective(model, ids, cfg, recurrent=recurrent, bank=bank)
        total = _gradients(output.loss, groups)
        auxiliary = _subtract(total, ntp)
        result[f"{name}_nmp_loss"] = float(output.metrics[f"{name}_nmp_loss"])
        for group, gradients in auxiliary.items():
            result[f"{name}_{group}_gradient_norm"] = _norm(gradients)
            result[f"{name}_{group}_vs_ntp_cosine"] = _cosine(
                ntp[group], gradients
            )
    return result


def _summary(records: Sequence[dict[str, float | None]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in sorted(records[0]):
        values = [float(record[key]) for record in records if record[key] is not None]
        result[key] = sum(values) / len(values) if values else None
    return result


def _candidate_weights(summary: dict[str, float | None]) -> dict[str, Any]:
    ntp = float(summary.get("ntp_pretrained_gradient_norm") or 0.0)
    result: dict[str, Any] = {}
    for fraction in (0.05, 0.10, 0.20):
        candidates: dict[str, float | None] = {}
        for objective in ("recurrent", "bank"):
            norm = float(
                summary.get(f"{objective}_pretrained_gradient_norm") or 0.0
            )
            if f"{objective}_nmp_loss" in summary:
                candidates[f"{objective}_weight"] = (
                    fraction * ntp / norm if norm > 0 else None
                )
        result[f"{fraction:g}"] = {
            "pretrained_gradient_fraction": fraction,
            **candidates,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=32)
    parser.add_argument("--head-warmup-batches", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batches <= 0 or args.head_warmup_batches <= 0 or args.start_index < 0:
        raise SystemExit("batches and head warm-up must be positive; start must be non-negative")

    config_path = _resolve(args.config)
    cfg = load_experiment_config(config_path)
    if cfg.nmp_detach_predictor_input:
        raise ValueError("calibration config must use coupled predictor inputs")
    checkpoint = _resolve(args.checkpoint or cfg.init_from or "")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    device = resolve_device(cfg.device)
    model = load_variant_from_config(cfg, device=device)
    provenance = load_model_weights(
        checkpoint,
        model=model,
        expected_experiment_config=cfg.to_dict(),
        allow_nmp_warm_start=cfg.allow_nmp_warm_start,
    )
    data = load_packed_dataset_for_experiment(
        cfg.data_dir,
        "train",
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
    )
    required = max(args.start_index + args.batches, args.head_warmup_batches)
    if required > len(data):
        raise ValueError(f"calibration needs {required} blocks; data has {len(data)}")
    groups = _parameter_groups(model)
    if not groups["head"] or not groups["pretrained"] or not groups["memory"]:
        raise RuntimeError("calibration requires head, pretrained, and memory parameters")

    def measure() -> list[dict[str, float | None]]:
        model.eval()
        records = []
        with torch.enable_grad():
            for index in range(args.start_index, args.start_index + args.batches):
                ids = data.batch([index], device=device)
                with autocast_context(device, cfg.autocast_dtype):
                    records.append(_measure(model, ids, cfg, groups))
        return records

    initial = _summary(measure())
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    head_ids = {id(parameter) for parameter in groups["head"]}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in head_ids)
    optimizer = torch.optim.AdamW(groups["head"], lr=cfg.added_lr, weight_decay=0.0)
    model.train()
    for index in range(args.head_warmup_batches):
        ids = data.batch([index], device=device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, cfg.autocast_dtype):
            output = _objective(
                model,
                ids,
                cfg,
                recurrent=float(model.recurrent_nmp_predictor is not None),
                bank=float(model.bank_nmp_predictor is not None),
            )
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(groups["head"], cfg.grad_clip)
        optimizer.step()
    for parameter, requires_grad in zip(
        model.parameters(), original_requires_grad, strict=True
    ):
        parameter.requires_grad_(requires_grad)
    post_warmup = _summary(measure())

    report = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_train_state": provenance["source_train_state"],
        "target_mode": cfg.nmp_target_mode,
        "batches": args.batches,
        "head_warmup_batches": args.head_warmup_batches,
        "initial": initial,
        "post_head_warmup": post_warmup,
        "candidate_weights": _candidate_weights(post_warmup),
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
