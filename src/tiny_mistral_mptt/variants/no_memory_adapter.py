"""Feedback-pass capacity control with no cross-pass memory."""

from collections.abc import Iterable

import torch
from torch import nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from .decoder import DecoderRun, run_memory_decoder
from .memory_modules import MemoryWriter, build_recurrent_mergers
from .multipass import MultiPassVariant


class NoMemoryAdapterVariant(MultiPassVariant):
    """Apply matched residual adapters only on passes after the first.

    The shared control projection is computed once from the residual at the
    first selected site and reused by later selected sites in the same pass.
    It never consumes a previous-pass state. Its parameter structure therefore
    matches the projected-residual feedback arm (one D-to-D writer plus one
    merger per site) without testing memory transfer.
    """

    variant_name = "no_memory_adapter"
    supports_cached_feedback = True

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        memory_layers: list[int] | tuple[int, ...],
        initialization_seed: int = 4242,
    ):
        super().__init__(backbone)
        if not isinstance(memory_layers, (list, tuple)) or not memory_layers:
            raise ValueError("no-memory adapter layers must be an explicit non-empty list")
        layers = tuple(sorted(int(layer) for layer in memory_layers))
        if (
            len(set(layers)) != len(layers)
            or layers[0] < 0
            or layers[-1] >= len(backbone.model.layers)
        ):
            raise ValueError("no-memory adapter layers must be unique valid decoder indices")
        self.memory_layers = layers
        hidden_size = int(backbone.config.hidden_size)
        self.writer = MemoryWriter(hidden_size)
        self.memory_mergers = build_recurrent_mergers(
            backbone.config,
            layers=layers,
            merger="projected_residual",
            initialization_seed=initialization_seed,
        )

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.writer.parameters()
        yield from self.memory_mergers.parameters()

    def _run_with_adapter(
        self,
        embeddings: torch.Tensor,
        *,
        past_key_values: tuple[LayerKVCache, ...] | None = None,
        use_cache: bool = False,
    ) -> DecoderRun:
        control_record: torch.Tensor | None = None

        def merge(layer: int, hidden: torch.Tensor) -> torch.Tensor:
            nonlocal control_record
            key = str(layer)
            if key not in self.memory_mergers:
                return hidden
            if control_record is None:
                control_record = self.writer(hidden)
            return self.memory_mergers[key](hidden, control_record)

        return run_memory_decoder(
            self.backbone,
            embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            after_attention=merge,
        )

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids, previous_hidden
        return self._run_with_adapter(token_embeddings).hidden_states

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        del input_ids, previous_hidden
        output = self._run_with_adapter(token_embeddings, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("cached no-memory adapter pass did not return KV state")
        return output.hidden_states, output.past_key_values

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        del feedback_memory, token
        output = self._run_with_adapter(
            token_embedding,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("cached no-memory adapter token did not return KV state")
        return output.hidden_states, output.past_key_values

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        return hidden_states[:, -1:, :].detach()

    def _append_feedback_memory(
        self,
        feedback_memory: torch.Tensor,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> torch.Tensor:
        del feedback_memory, token, position
        return new_hidden.detach()
