from __future__ import annotations

from collections.abc import Sequence

import torch

from tiny_mistral.modeling import MistralForCausalLM

from ..training.loss import normalize_pass_weights
from .base import ExperimentalVariant, TrainOutput


class VanillaVariant(ExperimentalVariant):
    variant_name = "vanilla"

    def __init__(self, backbone: MistralForCausalLM):
        super().__init__()
        self.backbone = backbone

    @property
    def config(self):
        return self.backbone.config

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        *,
        phase: str = "B",
        passes: int = 1,
        loss_weights: Sequence[float] | None = None,
    ) -> TrainOutput:
        if phase not in {"A", "B"}:
            raise ValueError("phase must be 'A' or 'B'")
        if phase == "A":
            raise ValueError("vanilla has no Phase-A training")
        if passes != 1:
            raise ValueError("vanilla variant supports exactly one pass")
        out = self.backbone(input_ids, labels=input_ids, use_cache=False)
        if out.loss is None:
            raise RuntimeError("vanilla backbone did not return a loss")
        weights = normalize_pass_weights(
            loss_weights,
            1,
            device=out.loss.device,
            dtype=out.loss.dtype,
        )
        return TrainOutput(
            loss=weights[0] * out.loss,
            pass_losses=(out.loss,),
            effective_passes=1,
            metrics={
                "pass_1_loss": float(out.loss.detach().cpu()),
                "pass_1_weight": float(weights[0].detach().cpu()),
            },
        )

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)
