from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from tiny_mistral.modeling import MistralForCausalLM, MistralRMSNorm

from ..feedback import HybridPassSource
from .bank_recurrent_hybrid import BankRecurrentHybridVariant


class BankAddHybridVariant(BankRecurrentHybridVariant):
    """Memory Attention plus a one-step MemoryAdd fast channel."""

    variant_name = "bank_add_hybrid"

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        memory_window: int = 32,
        memory_write_mode: str = "periodic",
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
        hidden_size = int(backbone.config.hidden_size)
        # Preserve the historical top-level names so existing BankAdd hybrid
        # checkpoints remain loadable after introducing the generic base.
        self.memory_norm = MistralRMSNorm(
            hidden_size, eps=float(backbone.config.rms_norm_eps)
        )
        with torch.random.fork_rng(devices=[]):
            self.memory_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.memory_projection.weight)

    def recurrent_added_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.memory_norm.parameters()
        yield from self.memory_projection.parameters()

    def memory_residual(
        self,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            if self.uses_memory_tokens:
                raise ValueError("memory-token hybrid residual requires input_ids")
            shifted = torch.zeros_like(previous_hidden)
            if previous_hidden.shape[1] > 1:
                shifted[:, 1:, :] = previous_hidden[:, :-1, :]
            source = shifted
        else:
            source = self._previous_ordinary_hidden(previous_hidden, input_ids)
        return self.memory_projection(self.memory_norm(source))

    def prepare_recurrent_inputs(
        self,
        token_embeddings: torch.Tensor,
        recurrent_source: torch.Tensor | None,
    ) -> torch.Tensor:
        if recurrent_source is None:
            return token_embeddings
        if recurrent_source.shape != token_embeddings.shape:
            raise ValueError("MemoryAdd source and token embeddings must share [B,T,D]")
        return token_embeddings + self.memory_projection(
            self.memory_norm(recurrent_source)
        )

    def _run_feedback_hidden_components(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        *,
        fast_hidden: torch.Tensor,
        bank_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.shape != fast_hidden.shape or token_embeddings.shape != bank_hidden.shape:
            raise ValueError("token embeddings and both feedback sources must share [B,T,D]")
        source = HybridPassSource(
            recurrent_hidden=fast_hidden,
            bank_hidden=bank_hidden,
        )
        return self._run_feedback_state(
            input_ids, token_embeddings, source
        ).hidden_states
