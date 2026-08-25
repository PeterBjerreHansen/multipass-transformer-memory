from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM, MistralRMSNorm

from .multipass import MultiPassVariant, shift_previous_hidden


class MemoryAddVariant(MultiPassVariant):
    """Single-state additive previous-pass feedback for pretrained TinyMistral.

    Pass 1 is exact vanilla TinyMistral. On every later pass, position ``t``
    receives a learned residual derived only from the previous pass's top-layer
    state at ``t-1``::

        x_t = e_t + W_M RMSNorm(h^{k-1}_{t-1})

    Position zero has no predecessor and therefore receives an exact zero
    residual. ``memory_projection`` is zero-initialized so all pass depths are
    exact vanilla at initialization. This is intentionally the current repo's
    one-state MemoryAdd control: it reuses the previous top hidden state
    directly rather than adding a separate learned memory-write head.
    """

    variant_name = "memory_add"
    supports_cached_feedback = True

    def __init__(self, backbone: MistralForCausalLM):
        super().__init__(backbone)
        hidden_size = int(backbone.config.hidden_size)
        self.memory_norm = MistralRMSNorm(
            hidden_size, eps=float(backbone.config.rms_norm_eps)
        )
        # nn.Linear performs a random default initialization in its constructor.
        # Fork the RNG even though the final projection is zero-initialized so
        # adding this variant never perturbs experiment/data RNG state.
        with torch.random.fork_rng(devices=[]):
            self.memory_projection = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.memory_projection.weight)

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().added_parameters()
        yield from self.memory_norm.parameters()
        yield from self.memory_projection.parameters()

    def memory_residual(self, previous_hidden: torch.Tensor) -> torch.Tensor:
        shifted = shift_previous_hidden(previous_hidden)
        return self.memory_projection(self.memory_norm(shifted))

    def feedback_inputs(
        self,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        if token_embeddings.shape != previous_hidden.shape:
            raise ValueError(
                "token_embeddings and previous_hidden must have identical [B,T,D] shape"
            )
        return token_embeddings + self.memory_residual(previous_hidden)

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        feedback = self.feedback_inputs(token_embeddings, previous_hidden)
        return self.backbone.model(
            inputs_embeds=feedback, use_cache=False
        ).last_hidden_state

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        del input_ids
        feedback = self.feedback_inputs(token_embeddings, previous_hidden)
        output = self.backbone.model(inputs_embeds=feedback, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("cached MemoryAdd prefill did not return KV state")
        return output.last_hidden_state, output.past_key_values

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if token_embedding.ndim != 3 or token_embedding.shape[1] != 1:
            raise ValueError("token_embedding must be [B,1,D]")
        if feedback_memory.shape != token_embedding.shape:
            raise ValueError("MemoryAdd cached feedback memory must be [B,1,D]")
        # feedback_memory already denotes h_{t-1}; do not right-shift it again.
        feedback = token_embedding + self.memory_projection(
            self.memory_norm(feedback_memory)
        )
        output = self.backbone.model(
            inputs_embeds=feedback,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("cached MemoryAdd token did not return KV state")
        return output.last_hidden_state, output.past_key_values

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError("hidden_states must be non-empty [B,T,D]")
        return hidden_states[:, -1:, :].detach()

    def _append_feedback_memory(
        self,
        feedback_memory: torch.Tensor,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> torch.Tensor:
        del token, position
        if feedback_memory.ndim != 3 or feedback_memory.shape[1] != 1:
            raise ValueError("MemoryAdd feedback memory must be [B,1,D]")
        if new_hidden.ndim != 3 or new_hidden.shape[1] != 1:
            raise ValueError("new_hidden must be [B,1,D]")
        if (
            feedback_memory.shape[0] != new_hidden.shape[0]
            or feedback_memory.shape[-1] != new_hidden.shape[-1]
        ):
            raise ValueError("feedback memory and new hidden shapes are incompatible")
        return new_hidden.detach()
