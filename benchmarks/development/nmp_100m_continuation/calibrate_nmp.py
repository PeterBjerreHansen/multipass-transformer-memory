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


def _objective(
    model,
    ids,
    cfg,
    *,
    passes: int,
    recurrent: float,
    bank: float,
):
    old_recurrent = model.recurrent_nmp_weight
    old_bank = model.bank_nmp_weight
    model.recurrent_nmp_weight = float(recurrent)
    model.bank_nmp_weight = float(bank)
    try:
        return model.compute_loss(
            ids,
            phase=cfg.phase,
            passes=passes,
            loss_weights=cfg.ntp_loss_weights_for_passes(passes),
            recurrent_nmp_loss_weights=cfg.recurrent_nmp_loss_weights_for_passes(
                passes
            ),
            bank_nmp_loss_weights=cfg.bank_nmp_loss_weights_for_passes(passes),
            nmp_weight_scale=1.0,
        )
    finally:
        model.recurrent_nmp_weight = old_recurrent
        model.bank_nmp_weight = old_bank


def _weighted_metric(outputs, probabilities: dict[int, float], name: str) -> float:
    return sum(
        float(probabilities[passes]) * float(output.metrics[name])
        for passes, output in outputs.items()
    )


def _weighted_gradients(
    gradients_by_pass: dict[int, dict[str, list[torch.Tensor]]],
    probabilities: dict[int, float],
) -> dict[str, list[torch.Tensor]]:
    first = next(iter(gradients_by_pass.values()))
    return {
        group: [
            sum(
                (
                    float(probabilities[passes]) * gradients[group][index]
                    for passes, gradients in gradients_by_pass.items()
                ),
                tensor.new_zeros(tensor.shape),
            )
            for index, tensor in enumerate(values)
        ]
        for group, values in first.items()
    }


def _measure(
    model,
    ids,
    cfg,
    groups,
    pass_probabilities: dict[int, float],
) -> dict[str, float | None]:
    ntp_outputs = {}
    ntp_gradients_by_pass = {}
    for passes in pass_probabilities:
        output = _objective(
            model,
            ids,
            cfg,
            passes=passes,
            recurrent=0.0,
            bank=0.0,
        )
        ntp_outputs[passes] = output
        ntp_gradients_by_pass[passes] = _gradients(output.loss, groups)
    ntp = _weighted_gradients(ntp_gradients_by_pass, pass_probabilities)
    result: dict[str, float | None] = {
        "ntp_loss": _weighted_metric(ntp_outputs, pass_probabilities, "ntp_loss"),
    }
    for group, gradients in ntp.items():
        result[f"ntp_{group}_gradient_norm"] = _norm(gradients)

    objectives = []
    if model.recurrent_nmp_predictor is not None:
        objectives.append(("recurrent", 1.0, 0.0))
    if model.bank_nmp_predictor is not None:
        objectives.append(("bank", 0.0, 1.0))
    for name, recurrent, bank in objectives:
        outputs = {}
        total_gradients_by_pass = {}
        for passes in pass_probabilities:
            output = _objective(
                model,
                ids,
                cfg,
                passes=passes,
                recurrent=recurrent,
                bank=bank,
            )
            outputs[passes] = output
            total_gradients_by_pass[passes] = _gradients(output.loss, groups)
        total = _weighted_gradients(
            total_gradients_by_pass, pass_probabilities
        )
        auxiliary = _subtract(total, ntp)
        result[f"{name}_nmp_loss"] = _weighted_metric(
            outputs, pass_probabilities, f"{name}_nmp_loss"
        )
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
    parser.add_argument(
        "--schedule-stage",
        type=int,
        default=0,
        help="zero-based pass-schedule stage whose exact mixture is calibrated",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        args.batches <= 0
        or args.head_warmup_batches <= 0
        or args.start_index < 0
        or args.schedule_stage < 0
    ):
        raise SystemExit("batches and head warm-up must be positive; start must be non-negative")

    config_path = _resolve(args.config)
    cfg = load_experiment_config(config_path)
    if cfg.nmp_detach_predictor_input:
        raise ValueError("calibration config must use coupled predictor inputs")
    stages = cfg.normalized_pass_schedule()
    if args.schedule_stage >= len(stages):
        raise ValueError(
            f"schedule stage {args.schedule_stage} does not exist; "
            f"config has {len(stages)} stage(s)"
        )
    pass_probabilities = {
        int(passes): float(probability)
        for passes, probability in stages[args.schedule_stage][
            "probabilities"
        ].items()
    }
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

    def measure(probabilities: dict[int, float]) -> list[dict[str, float | None]]:
        model.eval()
        records = []
        with torch.enable_grad():
            for index in range(args.start_index, args.start_index + args.batches):
                ids = data.batch([index], device=device)
                with autocast_context(device, cfg.autocast_dtype):
                    records.append(
                        _measure(model, ids, cfg, groups, probabilities)
                    )
        return records

    initial_by_pass = {
        str(passes): _summary(measure({passes: 1.0}))
        for passes in pass_probabilities
    }
    initial_mixture = _summary(measure(pass_probabilities))
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    head_ids = {id(parameter) for parameter in groups["head"]}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in head_ids)
    optimizer = torch.optim.AdamW(
        groups["head"], lr=cfg.nmp_predictor_lr, weight_decay=0.0
    )
    model.train()
    for index in range(args.head_warmup_batches):
        ids = data.batch([index], device=device)
        optimizer.zero_grad(set_to_none=True)
        for passes, probability in pass_probabilities.items():
            with autocast_context(device, cfg.autocast_dtype):
                output = _objective(
                    model,
                    ids,
                    cfg,
                    passes=passes,
                    recurrent=float(model.recurrent_nmp_predictor is not None),
                    bank=float(model.bank_nmp_predictor is not None),
                )
                loss = float(probability) * output.loss
            loss.backward()
        torch.nn.utils.clip_grad_norm_(groups["head"], cfg.grad_clip)
        optimizer.step()
    for parameter, requires_grad in zip(
        model.parameters(), original_requires_grad, strict=True
    ):
        parameter.requires_grad_(requires_grad)
    post_warmup_by_pass = {
        str(passes): _summary(measure({passes: 1.0}))
        for passes in pass_probabilities
    }
    post_warmup_mixture = _summary(measure(pass_probabilities))

    report = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_train_state": provenance["source_train_state"],
        "target_mode": cfg.nmp_target_mode,
        "schedule_stage": args.schedule_stage,
        "pass_probabilities": pass_probabilities,
        "batches": args.batches,
        "head_warmup_batches": args.head_warmup_batches,
        "head_warmup_learning_rate": cfg.nmp_predictor_lr,
        "initial_by_pass": initial_by_pass,
        "initial_mixture": initial_mixture,
        "post_head_warmup_by_pass": post_warmup_by_pass,
        "post_head_warmup_mixture": post_warmup_mixture,
        "candidate_weights": _candidate_weights(post_warmup_mixture),
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
