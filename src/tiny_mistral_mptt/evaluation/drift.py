from __future__ import annotations

import math
from typing import Any

import torch


def parameter_drift_summary(
    reference: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    *,
    added_names: set[str],
) -> dict[str, Any]:
    if set(reference) != set(current):
        raise ValueError("reference and current parameter names differ")
    groups: dict[str, dict[str, float | int]] = {
        "all": {},
        "backbone": {},
        "added": {},
    }
    tensors = []

    def accumulate(group: str, *, count: int, delta2: float, reference2: float,
                   current2: float, max_abs: float) -> None:
        values = groups[group]
        values["parameters"] = int(values.get("parameters", 0)) + count
        values["delta_squared_sum"] = float(values.get("delta_squared_sum", 0.0)) + delta2
        values["reference_squared_sum"] = float(values.get("reference_squared_sum", 0.0)) + reference2
        values["current_squared_sum"] = float(values.get("current_squared_sum", 0.0)) + current2
        values["max_absolute_delta"] = max(
            float(values.get("max_absolute_delta", 0.0)), max_abs
        )

    for name in sorted(reference):
        before = reference[name].detach().cpu().float()
        after = current[name].detach().cpu().float()
        if before.shape != after.shape:
            raise ValueError(f"parameter shape changed for {name}")
        delta = after - before
        count = delta.numel()
        delta2 = float(delta.square().sum().item())
        reference2 = float(before.square().sum().item())
        current2 = float(after.square().sum().item())
        max_abs = float(delta.abs().max().item()) if count else 0.0
        category = "added" if name in added_names else "backbone"
        accumulate(
            "all",
            count=count,
            delta2=delta2,
            reference2=reference2,
            current2=current2,
            max_abs=max_abs,
        )
        accumulate(
            category,
            count=count,
            delta2=delta2,
            reference2=reference2,
            current2=current2,
            max_abs=max_abs,
        )
        tensors.append(
            {
                "name": name,
                "category": category,
                "parameters": count,
                "rms_delta": math.sqrt(delta2 / count) if count else 0.0,
                "relative_l2_delta": (
                    math.sqrt(delta2 / reference2) if reference2 else None
                ),
                "max_absolute_delta": max_abs,
            }
        )

    rendered_groups: dict[str, Any] = {}
    for name, values in groups.items():
        count = int(values.get("parameters", 0))
        delta2 = float(values.get("delta_squared_sum", 0.0))
        reference2 = float(values.get("reference_squared_sum", 0.0))
        current2 = float(values.get("current_squared_sum", 0.0))
        rendered_groups[name] = {
            "parameters": count,
            "rms_delta": math.sqrt(delta2 / count) if count else None,
            "relative_l2_delta": (
                math.sqrt(delta2 / reference2) if reference2 else None
            ),
            "reference_rms": math.sqrt(reference2 / count) if count else None,
            "current_rms": math.sqrt(current2 / count) if count else None,
            "max_absolute_delta": values.get("max_absolute_delta"),
        }
    return {"groups": rendered_groups, "tensors": tensors}
