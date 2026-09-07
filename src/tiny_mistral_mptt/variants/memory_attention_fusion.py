from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


MEMORY_ATTENTION_FUSIONS = {
    "residual",
    "destination_gated",
    "attention_gated",
    "dual_gated",
}


@dataclass(frozen=True)
class FusionGates:
    """Optional feature-wise gates produced by one attention-fusion site."""

    alpha: torch.Tensor | None = None
    beta: torch.Tensor | None = None


class _FusionGateController(nn.Module):
    """Shared controller form for the single- and dual-gate ablations."""

    def __init__(
        self,
        hidden_size: int,
        *,
        controller_hidden_size: int,
        gate_names: tuple[str, ...],
        initialization_seed: int,
    ):
        super().__init__()
        if controller_hidden_size < 1:
            raise ValueError("controller_hidden_size must be positive")
        if not gate_names or not set(gate_names) <= {"alpha", "beta"}:
            raise ValueError("gate_names must contain alpha, beta, or both")
        if len(gate_names) != len(set(gate_names)):
            raise ValueError("gate_names must be unique")
        self.hidden_size = int(hidden_size)
        self.gate_names = gate_names
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.input_norm = nn.LayerNorm(2 * hidden_size)
            self.hidden_1 = nn.Linear(2 * hidden_size, controller_hidden_size)
            self.hidden_2 = nn.Linear(controller_hidden_size, controller_hidden_size)
            self.output = nn.Linear(
                controller_hidden_size, len(gate_names) * hidden_size
            )
            nn.init.xavier_uniform_(self.hidden_1.weight)
            nn.init.zeros_(self.hidden_1.bias)
            nn.init.xavier_uniform_(self.hidden_2.weight)
            nn.init.zeros_(self.hidden_2.bias)
            nn.init.zeros_(self.output.weight)
            for gate_index, gate_name in enumerate(gate_names):
                probability = 0.1 if gate_name == "alpha" else 0.9
                logit = math.log(probability / (1.0 - probability))
                start = gate_index * hidden_size
                nn.init.constant_(self.output.bias[start : start + hidden_size], logit)

    def forward(
        self, attention: torch.Tensor, destination: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        features = torch.cat((attention, destination), dim=-1)
        hidden = F.gelu(self.hidden_1(self.input_norm(features)))
        hidden = F.gelu(self.hidden_2(hidden))
        values = torch.sigmoid(self.output(hidden)).chunk(
            len(self.gate_names), dim=-1
        )
        return dict(zip(self.gate_names, values, strict=True))


class MemoryAttentionFusion(nn.Module):
    """Fuse a raw attention read into one destination stream.

    All modes share the same call boundary and exact no-memory bypass. Gated
    modes differ only in whether they learn a feature-wise gate on the raw
    attention contribution, the destination contribution, or both.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        mode: str,
        controller_hidden_size: int | None,
        initialization_seed: int,
    ):
        super().__init__()
        if mode not in MEMORY_ATTENTION_FUSIONS:
            raise ValueError(
                "mode must be residual, destination_gated, attention_gated, "
                "or dual_gated"
            )
        if mode == "residual":
            if controller_hidden_size is not None:
                raise ValueError(
                    "controller_hidden_size is not used by residual fusion"
                )
            gate_names: tuple[str, ...] = ()
        else:
            if controller_hidden_size is None or controller_hidden_size < 1:
                raise ValueError(
                    "gated fusion requires a positive controller_hidden_size"
                )
            gate_names = {
                "destination_gated": ("beta",),
                "attention_gated": ("alpha",),
                "dual_gated": ("alpha", "beta"),
            }[mode]
        self.mode = mode
        self.controller = (
            None
            if not gate_names
            else _FusionGateController(
                hidden_size,
                controller_hidden_size=int(controller_hidden_size),
                gate_names=gate_names,
                initialization_seed=initialization_seed,
            )
        )

    def gate_values(
        self, attention: torch.Tensor, destination: torch.Tensor
    ) -> FusionGates:
        if attention.shape != destination.shape:
            raise ValueError("attention and destination must have the same shape")
        if self.controller is None:
            return FusionGates()
        values = self.controller(attention, destination)
        return FusionGates(alpha=values.get("alpha"), beta=values.get("beta"))

    def forward(
        self,
        destination: torch.Tensor,
        attention: torch.Tensor,
        *,
        available: torch.Tensor,
    ) -> torch.Tensor:
        if destination.ndim != 3 or attention.shape != destination.shape:
            raise ValueError("destination and attention must be matching [B,T,D]")
        if available.shape != destination.shape[:2] or available.dtype != torch.bool:
            raise ValueError("available must be boolean [B,T]")

        gates = self.gate_values(attention, destination)
        attention_contribution = (
            attention if gates.alpha is None else gates.alpha * attention
        )
        destination_contribution = (
            destination if gates.beta is None else gates.beta * destination
        )
        fused = destination_contribution + attention_contribution
        return torch.where(available[:, :, None], fused, destination)
