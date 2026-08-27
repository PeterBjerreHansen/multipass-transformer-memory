from __future__ import annotations

from collections.abc import Iterable
import math

import torch
import torch.nn as nn

from tiny_mistral.modeling import MistralForCausalLM

from .recirculation import _AdaptiveRecirculationController
from .bank_recurrent_hybrid import MemoryAttentionRecurrentHybridVariant


class RecirculationStridedMemoryAttentionVariant(MemoryAttentionRecurrentHybridVariant):
    """Sparse Memory Attention plus fixed or adaptive layer recirculation."""

    variant_name = "recirculation_strided_memory_attention"

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        source_layer: int,
        destination_layer: int,
        alpha: float = 0.1,
        mode: str = "adaptive",
        memory_window: int = 32,
        memory_write_mode: str = "strided",
        memory_write_stride: int = 8,
        memory_token_visibility: str = "visible",
        memory_layers: str | list[int] = "all",
        memory_position_encoding: str = "rope",
        initialization_seed: int = 4242,
    ):
        super().__init__(
            backbone,
            memory_window=memory_window,
            memory_write_mode=memory_write_mode,
            memory_write_stride=memory_write_stride,
            memory_token_visibility=memory_token_visibility,
            memory_layers=memory_layers,
            memory_position_encoding=memory_position_encoding,
            initialization_seed=initialization_seed,
        )
        layer_count = len(backbone.model.layers)
        if not (0 <= destination_layer < source_layer < layer_count):
            raise ValueError(
                "require 0 <= destination_layer < source_layer < num_layers"
            )
        if not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be finite in [0,1]")
        if mode not in {"fixed", "adaptive"}:
            raise ValueError("recirculation mode must be 'fixed' or 'adaptive'")
        self.source_layer = int(source_layer)
        self.destination_layer = int(destination_layer)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.adaptive_controller: _AdaptiveRecirculationController | None = None
        if self.mode == "adaptive":
            self.adaptive_controller = _AdaptiveRecirculationController(
                int(backbone.config.hidden_size),
                initial_alpha=self.alpha,
                initialization_seed=initialization_seed,
            )

    def recurrent_added_parameters(self) -> Iterable[nn.Parameter]:
        if self.adaptive_controller is not None:
            yield from self.adaptive_controller.parameters()

    @property
    def recurrent_capture_layer(self) -> int:
        return self.source_layer

    def prepare_recurrent_inputs(
        self,
        token_embeddings: torch.Tensor,
        recurrent_source: torch.Tensor | None,
    ) -> torch.Tensor:
        del recurrent_source
        return token_embeddings

    @staticmethod
    def _norm_match(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        source_norm = torch.linalg.vector_norm(
            source.float(), ord=2, dim=-1, keepdim=True
        )
        destination_norm = torch.linalg.vector_norm(
            destination.float(), ord=2, dim=-1, keepdim=True
        )
        scale = (destination_norm / source_norm.clamp_min(1e-12)).to(source.dtype)
        return source * scale

    def _mix(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        *,
        valid_feedback: torch.Tensor | None,
    ) -> torch.Tensor:
        if source.shape != destination.shape:
            raise ValueError("recirculation source and destination shapes differ")
        matched = self._norm_match(source, destination)
        if self.adaptive_controller is None:
            candidate = self.alpha * matched + (1.0 - self.alpha) * destination
        else:
            alpha, beta = self.adaptive_controller(source, destination)
            candidate = alpha * matched + beta * destination
        if valid_feedback is None:
            return candidate
        return torch.where(valid_feedback[..., None], candidate, destination)

    def apply_recurrent_layer(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        recurrent_source: torch.Tensor | None,
        valid_feedback: torch.Tensor | None,
    ) -> torch.Tensor:
        if layer_index != self.destination_layer or recurrent_source is None:
            return hidden_states
        return self._mix(
            recurrent_source,
            hidden_states,
            valid_feedback=valid_feedback,
        )


BankRecirculationHybridVariant = RecirculationStridedMemoryAttentionVariant

__all__ = [
    "RecirculationStridedMemoryAttentionVariant",
    "BankRecirculationHybridVariant",
]
