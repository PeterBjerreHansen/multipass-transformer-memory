from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from ..feedback import HybridFeedbackState, HybridPassSource, MemoryAttentionState
from .memory_modules import build_recurrent_mergers
from .multipass import HiddenRun
from .memory_attention import MemoryAttentionBatch, MemoryAttentionVariant


class MemoryAttentionRecurrentHybridVariant(MemoryAttentionVariant):
    """Optional late-memory recurrence alongside configurable Memory Attention.

    Both channels consume final normalized source states through one shared
    writer. Attention reads run before recurrent mergers at overlapping layers,
    after self-attention and before the MLP. MEM records do not advance the
    preceding-ordinary-token recurrent state.
    """

    def __init__(
        self, backbone: MistralForCausalLM, *,
        recurrent_merger: str, recurrent_layers: list[int] | tuple[int, ...],
        recurrent_controller_hidden_size: int | None = None,
        initialization_seed: int = 4242, **attention_settings,
    ):
        super().__init__(backbone, initialization_seed=initialization_seed, **attention_settings)
        self.recurrent_merger = recurrent_merger
        self.memory_mergers = build_recurrent_mergers(
            backbone.config, layers=recurrent_layers, merger=recurrent_merger,
            controller_hidden_size=recurrent_controller_hidden_size,
            initialization_seed=initialization_seed,
        )
        self.recurrent_layers = tuple(int(key) for key in self.memory_mergers)

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().added_parameters()
        yield from self.memory_mergers.parameters()

    def _previous_ordinary_hidden_with_valid(
        self,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_hidden.ndim != 3 or input_ids.shape != previous_hidden.shape[:2]:
            raise ValueError("previous_hidden/input_ids must align as [B,T,D]/[B,T]")
        bsz, seq_len = input_ids.shape
        if not self.uses_memory_tokens:
            shifted = torch.zeros_like(previous_hidden)
            if seq_len > 1:
                shifted[:, 1:, :] = previous_hidden[:, :-1, :]
            valid = torch.ones((bsz, seq_len), dtype=torch.bool, device=input_ids.device)
            valid[:, 0] = False
            return shifted, valid

        ordinary = ~self.memory_token_mask(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        candidates = torch.where(
            ordinary,
            positions[None, :].expand(bsz, -1),
            torch.full((bsz, seq_len), -1, device=input_ids.device, dtype=torch.long),
        )
        inclusive = torch.cummax(candidates, dim=1).values
        strict = torch.cat(
            (
                torch.full((bsz, 1), -1, device=input_ids.device, dtype=torch.long),
                inclusive[:, :-1],
            ),
            dim=1,
        )
        safe = strict.clamp_min(0)
        gathered = previous_hidden.gather(
            1, safe[:, :, None].expand(-1, -1, previous_hidden.shape[-1])
        )
        valid = strict.ge(0)
        return (
            torch.where(valid[:, :, None], gathered, torch.zeros_like(gathered)),
            valid,
        )

    def _previous_ordinary_hidden(
        self,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Strictly previous ordinary recurrence source at each physical position."""
        source, _ = self._previous_ordinary_hidden_with_valid(previous_hidden, input_ids)
        return source

    @staticmethod
    def _coerce_pass_source(
        previous_source: HybridPassSource | torch.Tensor,
    ) -> HybridPassSource:
        if isinstance(previous_source, HybridPassSource):
            return previous_source
        if isinstance(previous_source, torch.Tensor):
            # Compatibility for direct callers of the historical hidden-only
            # hybrid hooks: the same stream supplies both channels.
            return HybridPassSource(previous_source, previous_source)
        raise TypeError("hybrid feedback requires HybridPassSource")

    def _run_hybrid_core(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        *,
        recurrent_source: torch.Tensor | None,
        memory: MemoryAttentionBatch | MemoryAttentionState | None,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        full_sequence_feedback: bool,
        query_position_ids: torch.Tensor | None = None,
    ) -> HiddenRun:
        aligned_source = recurrent_source
        valid_feedback: torch.Tensor | None = None
        if recurrent_source is not None:
            if full_sequence_feedback:
                aligned_source, valid_feedback = self._previous_ordinary_hidden_with_valid(
                    self.writer(recurrent_source), input_ids
                )
            else:
                if recurrent_source.shape != token_embeddings.shape:
                    raise ValueError("cached recurrent source must match [B,1,D] token input")
                valid_feedback = torch.ones(
                    token_embeddings.shape[:2],
                    dtype=torch.bool,
                    device=token_embeddings.device,
                )

        def merge(layer_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
            key = str(layer_index)
            if aligned_source is None or key not in self.memory_mergers:
                return hidden_states
            candidate = self.memory_mergers[key](hidden_states, aligned_source)
            if valid_feedback is None:
                return candidate
            return torch.where(valid_feedback[..., None], candidate, hidden_states)

        core = self._run_attention_core(
            token_embeddings, memory, past_key_values=past_key_values,
            use_cache=use_cache, self_attention_mask=self.self_attention_key_mask(input_ids),
            query_position_ids=query_position_ids, after_memory_attention=merge,
        )
        return HiddenRun(
            hidden_states=core.hidden_states,
            feedback_source=HybridPassSource(core.hidden_states, core.hidden_states),
            past_key_values=core.past_key_values,
        )

    def _run_first_state(self, input_ids: torch.Tensor) -> HiddenRun:
        return self._run_hybrid_core(
            input_ids,
            self.input_embeddings(input_ids),
            recurrent_source=None,
            memory=None,
            past_key_values=None,
            use_cache=False,
            full_sequence_feedback=False,
        )

    def _run_feedback_state(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_source: HybridPassSource | torch.Tensor,
    ) -> HiddenRun:
        source = self._coerce_pass_source(previous_source)
        return self._run_hybrid_core(
            input_ids,
            token_embeddings,
            recurrent_source=source.recurrent_hidden,
            memory=self.build_memory(source.memory_attention_hidden, input_ids),
            past_key_values=None,
            use_cache=False,
            full_sequence_feedback=True,
        )

    def _run_first_state_cached(self, input_ids: torch.Tensor) -> HiddenRun:
        return self._run_hybrid_core(
            input_ids,
            self.input_embeddings(input_ids),
            recurrent_source=None,
            memory=None,
            past_key_values=None,
            use_cache=True,
            full_sequence_feedback=False,
        )

    def _run_feedback_state_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_source: HybridPassSource | torch.Tensor,
    ) -> HiddenRun:
        source = self._coerce_pass_source(previous_source)
        return self._run_hybrid_core(
            input_ids,
            token_embeddings,
            recurrent_source=source.recurrent_hidden,
            memory=self.build_memory(source.memory_attention_hidden, input_ids),
            past_key_values=None,
            use_cache=True,
            full_sequence_feedback=True,
        )

    def _run_first_token_state_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> HiddenRun:
        return self._run_hybrid_core(
            input_ids,
            self.input_embeddings(input_ids),
            recurrent_source=None,
            memory=None,
            past_key_values=past_key_values,
            use_cache=True,
            full_sequence_feedback=False,
        )

    def _run_feedback_token_state_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: HybridFeedbackState,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> HiddenRun:
        if not isinstance(feedback_memory, HybridFeedbackState):
            raise TypeError("hybrid cached feedback requires HybridFeedbackState")
        if token is None:
            raise ValueError("hybrid cached feedback requires current token ID")
        return self._run_hybrid_core(
            token,
            token_embedding,
            recurrent_source=feedback_memory.recurrent_memory,
            memory=feedback_memory.memory_attention,
            past_key_values=past_key_values,
            use_cache=True,
            full_sequence_feedback=False,
            query_position_ids=self._cached_query_positions(feedback_memory.memory_attention, token),
        )

    # Hidden-only compatibility hooks.
    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self._run_feedback_state(
            input_ids, token_embeddings, previous_hidden
        ).hidden_states

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        run = self._run_feedback_state_cached(input_ids, token_embeddings, previous_hidden)
        if run.past_key_values is None:
            raise RuntimeError("cached hybrid pass did not return KV state")
        return run.hidden_states, run.past_key_values

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: HybridFeedbackState,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        run = self._run_feedback_token_state_cached(
            token_embedding, feedback_memory, past_key_values, token=token
        )
        if run.past_key_values is None:
            raise RuntimeError("cached hybrid token did not return KV state")
        return run.hidden_states, run.past_key_values

    def _last_ordinary_hidden(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.uses_memory_tokens:
            return hidden_states[:, -1:, :].detach()
        if input_ids is None:
            raise ValueError("memory-token hybrid state requires input_ids")
        ordinary = ~self.memory_token_mask(input_ids)
        if bool((ordinary.sum(dim=1) == 0).any()):
            raise ValueError("each sequence needs at least one ordinary token for hybrid state")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        last = torch.where(ordinary, positions, torch.full_like(positions, -1)).max(dim=1).values
        gathered = hidden_states.gather(
            1, last[:, None, None].expand(-1, 1, hidden_states.shape[-1])
        )
        return gathered.detach()

    def _feedback_memory_from_hidden(
        self,
        feedback_source: HybridPassSource | torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> HybridFeedbackState:
        source = self._coerce_pass_source(feedback_source)
        memory = super()._feedback_memory_from_hidden(
            source.memory_attention_hidden, input_ids=input_ids
        )
        return HybridFeedbackState(
            recurrent_memory=self.writer(self._last_ordinary_hidden(
                source.recurrent_hidden, input_ids
            )).detach(),
            memory_attention=memory,
        )

    def _append_feedback_memory(
        self,
        feedback_memory: HybridFeedbackState,
        new_feedback_source: HybridPassSource | torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> HybridFeedbackState:
        if not isinstance(feedback_memory, HybridFeedbackState):
            raise TypeError("hybrid feedback requires HybridFeedbackState")
        source = self._coerce_pass_source(new_feedback_source)
        memory = super()._append_feedback_memory(
            feedback_memory.memory_attention,
            source.memory_attention_hidden,
            token=token,
            position=position,
        )
        if self.uses_memory_tokens:
            if token is None:
                raise ValueError("memory-token hybrid update requires current token")
            is_mem = self.memory_token_mask(token)[:, 0]
            recurrent = torch.where(
                is_mem[:, None, None],
                feedback_memory.recurrent_memory,
                self.writer(source.recurrent_hidden).detach(),
            )
        else:
            recurrent = self.writer(source.recurrent_hidden).detach()
        return HybridFeedbackState(recurrent_memory=recurrent, memory_attention=memory)


__all__ = [
    "MemoryAttentionRecurrentHybridVariant",
]
