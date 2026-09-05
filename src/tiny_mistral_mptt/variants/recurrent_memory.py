"""Single-record recurrent memory with a shared late writer and selectable merger."""

from collections.abc import Iterable

import torch
from torch import nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from .decoder import DecoderRun, run_memory_decoder
from .memory_modules import MemoryWriter, build_recurrent_mergers
from .multipass import MultiPassVariant, shift_previous_hidden


class RecurrentMemoryVariant(MultiPassVariant):
    """Read only the preceding token's emitted memory from the preceding pass.

    Like Memory Attention, the writer consumes the final normalized backbone
    state and mergers run after self-attention, before the selected layer's MLP.
    Memory is written when consumed so the Phase-A no-grad first backbone pass
    does not detach the trainable writer. Cached paths write each new record once.
    There is no accumulated state or same-token replay policy.
    """

    variant_name = "recurrent_memory"
    supports_cached_feedback = True

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        memory_layers: list[int] | tuple[int, ...],
        merger: str,
        controller_hidden_size: int | None = None,
        initialization_seed: int = 4242,
    ):
        super().__init__(backbone)
        if merger not in {"projected_residual", "recirculation"}:
            raise ValueError("recurrent merger must be projected_residual or recirculation")
        if not isinstance(memory_layers, (list, tuple)) or not memory_layers:
            raise ValueError("recurrent memory_layers must be an explicit non-empty list")
        layers = tuple(sorted(int(layer) for layer in memory_layers))
        if (
            len(set(layers)) != len(layers)
            or layers[0] < 0
            or layers[-1] >= len(backbone.model.layers)
        ):
            raise ValueError("recurrent memory_layers must be unique valid decoder indices")
        self.memory_layers = layers
        self.memory_window = 1
        self.recurrent_merger = merger
        hidden_size = int(backbone.config.hidden_size)
        self.writer = MemoryWriter(hidden_size)
        self.memory_mergers = build_recurrent_mergers(
            backbone.config,
            layers=layers,
            merger=merger,
            controller_hidden_size=controller_hidden_size,
            initialization_seed=initialization_seed,
        )

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.writer.parameters()
        yield from self.memory_mergers.parameters()

    def _run_with_memory(
        self,
        embeddings: torch.Tensor,
        memory: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
        past_key_values: tuple[LayerKVCache, ...] | None = None,
        use_cache: bool = False,
    ) -> DecoderRun:
        if memory.shape != embeddings.shape:
            raise ValueError("recurrent memory and embeddings must share [B,T,D]")

        def merge(layer: int, hidden: torch.Tensor) -> torch.Tensor:
            key = str(layer)
            if key not in self.memory_mergers:
                return hidden
            candidate = self.memory_mergers[key](hidden, memory)
            if valid is None:
                return candidate
            return torch.where(valid[..., None], candidate, hidden)

        return run_memory_decoder(
            self.backbone,
            embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            after_attention=merge,
        )

    def _parallel_memory(
        self, previous_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory = shift_previous_hidden(self.writer(previous_hidden))
        valid = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        valid[:, 0] = False
        return memory, valid

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids
        memory, valid = self._parallel_memory(previous_hidden)
        return self._run_with_memory(token_embeddings, memory, valid=valid).hidden_states

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        del input_ids
        memory, valid = self._parallel_memory(previous_hidden)
        output = self._run_with_memory(token_embeddings, memory, valid=valid, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("cached recurrent pass did not return KV state")
        return output.hidden_states, output.past_key_values

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        del token
        if token_embedding.ndim != 3 or token_embedding.shape[1] != 1:
            raise ValueError("cached recurrent input must be [B,1,D]")
        output = self._run_with_memory(
            token_embedding, feedback_memory, past_key_values=past_key_values, use_cache=True
        )
        if output.past_key_values is None:
            raise RuntimeError("cached recurrent token did not return KV state")
        return output.hidden_states, output.past_key_values

    def _feedback_memory_from_hidden(
        self, hidden_states: torch.Tensor, *, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        del input_ids
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError("recurrent source must be non-empty [B,T,D]")
        return self.writer(hidden_states[:, -1:, :]).detach()

    def _append_feedback_memory(
        self,
        feedback_memory: torch.Tensor,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> torch.Tensor:
        del token, position
        if (
            feedback_memory.shape != new_hidden.shape
            or new_hidden.ndim != 3
            or new_hidden.shape[1] != 1
        ):
            raise ValueError("cached recurrent memory/source must share [B,1,D]")
        return self.writer(new_hidden).detach()
