from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
import torch.nn as nn

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from ..feedback import HybridFeedbackState, HybridPassSource, BankState
from .multipass import HiddenRun
from .bank import BankBatch, BankCoreRun, BankVariant


class BankRecurrentHybridVariant(BankVariant, ABC):
    """Composable Memory Attention plus fast-recurrence MPT base class.

    A recurrent mechanism supplies three small hooks: how it changes the input,
    whether it changes an internal decoder layer, and which layer output becomes
    the next recurrent source. The base owns memory-attention reads/writes, causal source
    alignment, multipass plumbing, and exact/recurrent cached inference.

    This makes the experimental axis explicit: MemoryAdd and recirculation use
    the same Memory Attention implementation and differ only in their fast routing rule.
    """

    supports_recurrent_nmp = True

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

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().added_parameters()
        yield from self.recurrent_added_parameters()

    def recurrent_added_parameters(self) -> Iterable[nn.Parameter]:
        return ()

    @property
    def recurrent_capture_layer(self) -> int | None:
        """Decoder layer whose output is fed to the next pass/token.

        ``None`` selects the final normalized top state, as used by MemoryAdd.
        """
        return None

    @abstractmethod
    def prepare_recurrent_inputs(
        self,
        token_embeddings: torch.Tensor,
        recurrent_source: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply recurrence that belongs before decoder layer zero."""

    def apply_recurrent_layer(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        recurrent_source: torch.Tensor | None,
        valid_feedback: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply recurrence after a decoder layer; default is a no-op."""
        del layer_index, recurrent_source, valid_feedback
        return hidden_states

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
        bank: BankBatch | BankState | None,
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
                    recurrent_source, input_ids
                )
            else:
                if recurrent_source.shape != token_embeddings.shape:
                    raise ValueError("cached recurrent source must match [B,1,D] token input")
                valid_feedback = torch.ones(
                    token_embeddings.shape[:2],
                    dtype=torch.bool,
                    device=token_embeddings.device,
                )

        feedback_inputs = self.prepare_recurrent_inputs(token_embeddings, aligned_source)

        def post_layer(layer_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.apply_recurrent_layer(
                layer_index,
                hidden_states,
                aligned_source,
                valid_feedback,
            )

        core = self._run_bank_core(
            feedback_inputs,
            bank,
            past_key_values=past_key_values,
            use_cache=use_cache,
            self_attention_mask=self.self_attention_key_mask(input_ids),
            query_position_ids=query_position_ids,
            post_layer=post_layer,
            capture_layer=self.recurrent_capture_layer,
        )
        next_recurrent = self._recurrent_source_from_core(core)
        return HiddenRun(
            hidden_states=core.hidden_states,
            feedback_source=HybridPassSource(next_recurrent, core.hidden_states),
            past_key_values=core.past_key_values,
        )

    def _recurrent_source_from_core(self, core: BankCoreRun) -> torch.Tensor:
        if self.recurrent_capture_layer is None:
            return core.hidden_states
        if core.captured_hidden is None:
            raise RuntimeError("hybrid recurrent source layer was not captured")
        return core.captured_hidden

    def _run_first_state(self, input_ids: torch.Tensor) -> HiddenRun:
        return self._run_hybrid_core(
            input_ids,
            self.input_embeddings(input_ids),
            recurrent_source=None,
            bank=None,
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
            bank=self.build_bank(source.bank_hidden, input_ids),
            past_key_values=None,
            use_cache=False,
            full_sequence_feedback=True,
        )

    def _run_first_state_cached(self, input_ids: torch.Tensor) -> HiddenRun:
        return self._run_hybrid_core(
            input_ids,
            self.input_embeddings(input_ids),
            recurrent_source=None,
            bank=None,
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
            bank=self.build_bank(source.bank_hidden, input_ids),
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
            bank=None,
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
            recurrent_source=feedback_memory.recurrent_hidden,
            bank=feedback_memory.bank,
            past_key_values=past_key_values,
            use_cache=True,
            full_sequence_feedback=False,
            query_position_ids=self._cached_query_positions(feedback_memory.bank, token),
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
        bank = super()._feedback_memory_from_hidden(
            source.bank_hidden, input_ids=input_ids
        )
        return HybridFeedbackState(
            fast_hidden=self._last_ordinary_hidden(
                source.recurrent_hidden, input_ids
            ),
            bank=bank,
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
        bank = super()._append_feedback_memory(
            feedback_memory.bank,
            source.bank_hidden,
            token=token,
            position=position,
        )
        if self.uses_memory_tokens:
            if token is None:
                raise ValueError("memory-token hybrid update requires current token")
            is_mem = self.memory_token_mask(token)[:, 0]
            recurrent = torch.where(
                is_mem[:, None, None],
                feedback_memory.recurrent_hidden,
                source.recurrent_hidden.detach(),
            )
        else:
            recurrent = source.recurrent_hidden.detach()
        return HybridFeedbackState(fast_hidden=recurrent, bank=bank)
