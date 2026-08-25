from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn


@dataclass
class TrainOutput:
    loss: torch.Tensor
    pass_losses: tuple[torch.Tensor, ...]
    effective_passes: int
    metrics: dict[str, float] = field(default_factory=dict)


class ExperimentalVariant(nn.Module):
    """Small trainer-facing contract shared by vanilla and multipass variants."""

    variant_name: str

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        *,
        phase: str = "B",
        passes: int = 1,
        loss_weights: Sequence[float] | None = None,
    ) -> TrainOutput:
        raise NotImplementedError

    def added_parameters(self) -> Iterable[nn.Parameter]:
        """Parameters absent from the validated vanilla backbone."""
        return ()

    def control_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Architecture control positions excluded from linguistic-token accounting."""
        return torch.zeros_like(input_ids, dtype=torch.bool)

    def linguistic_token_count(self, input_ids: torch.Tensor) -> int:
        return int((~self.control_token_mask(input_ids)).sum().item())

    def build_lm_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Default ordinary next-token targets aligned to prediction positions."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        labels = torch.full_like(input_ids, -100)
        if input_ids.shape[1] > 1:
            labels[:, :-1] = input_ids[:, 1:]
        return labels

    def set_phase(self, phase: str) -> None:
        if phase not in {"A", "B"}:
            raise ValueError("phase must be 'A' or 'B'")
        if phase == "A":
            added_ids = {id(parameter) for parameter in self.added_parameters()}
            for parameter in self.parameters():
                parameter.requires_grad_(id(parameter) in added_ids)
        else:
            for parameter in self.parameters():
                parameter.requires_grad_(True)
