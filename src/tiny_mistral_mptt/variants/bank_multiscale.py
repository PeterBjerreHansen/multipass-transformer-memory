from __future__ import annotations

import torch

from tiny_mistral.attention.multiresolution import retained_multiresolution_indices
from tiny_mistral.modeling import MistralForCausalLM

from ..feedback import MemoryAttentionState
from .bank import MemoryAttentionBatch, MemoryAttentionReader, MemoryAttentionVariant


class MultiscaleMemoryAttentionVariant(MemoryAttentionVariant):
    """Multiscale Memory Attention with dense-recent and sparse-old retention.

    Every previous-pass top state is written through the shared memory writer.
    Each query reads one concatenated union: the preceding ``D`` states and the
    last ``S`` fixed-stride states strictly older than that dense region. A
    The memory-attention reader applies one Q/K/V projection set and one softmax
    to the union.
    """

    variant_name = "memory_attention_multiscale"

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        memory_dense_window: int = 32,
        memory_sparse_window: int = 32,
        memory_sparse_stride: int = 32,
        memory_layers: str | list[int] = "all",
        memory_position_encoding: str = "rope",
        initialization_seed: int = 4242,
    ):
        dense_window = int(memory_dense_window)
        sparse_window = int(memory_sparse_window)
        sparse_stride = int(memory_sparse_stride)
        if dense_window < 0 or sparse_window < 0:
            raise ValueError("Multiscale Memory Attention windows must be non-negative")
        if dense_window + sparse_window <= 0:
            raise ValueError("Multiscale Memory Attention requires at least one non-zero window")
        if sparse_stride <= 0:
            raise ValueError("memory_sparse_stride must be positive")

        super().__init__(
            backbone,
            memory_window=dense_window + sparse_window,
            memory_write_mode="dense",
            memory_write_stride=1,
            memory_token_visibility="visible",
            memory_layers=memory_layers,
            memory_position_encoding=memory_position_encoding,
            initialization_seed=initialization_seed,
        )
        self.memory_dense_window = dense_window
        self.memory_sparse_window = sparse_window
        self.memory_sparse_stride = sparse_stride

    def _full_bank_memory_delta(
        self,
        memory_reader: MemoryAttentionReader,
        hidden_states: torch.Tensor,
        bank: MemoryAttentionBatch,
        *,
        query_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        return memory_reader.forward_multiscale_bank(
            hidden_states,
            bank.memories,
            memory_mask=bank.valid,
            dense_window=self.memory_dense_window,
            sparse_stride=self.memory_sparse_stride,
            sparse_window=self.memory_sparse_window,
            query_position_ids=query_position_ids,
            memory_position_ids=bank.memory_positions,
        )

    def _selection(
        self,
        positions: torch.Tensor,
        valid: torch.Tensor,
        next_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return retained_multiresolution_indices(
            positions,
            valid,
            next_positions,
            recent_window=self.memory_dense_window,
            sparse_stride=self.memory_sparse_stride,
            sparse_window=self.memory_sparse_window,
        )

    @staticmethod
    def _gather_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.gather(
            values,
            1,
            indices[:, :, None].expand(-1, -1, values.shape[-1]),
        )

    def _state_from_bank_batch(self, bank: MemoryAttentionBatch) -> MemoryAttentionState:
        next_positions = bank.query_positions[:, -1] + 1
        selection, selected_valid = self._selection(
            bank.memory_positions,
            bank.valid,
            next_positions,
        )
        memories = self._gather_rows(bank.memories, selection)
        positions = torch.gather(bank.memory_positions, 1, selection)
        return self._project_state(
            memories,
            selected_valid,
            positions,
            next_positions,
        )

    def _append_feedback_memory(
        self,
        feedback_memory: MemoryAttentionState,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> MemoryAttentionState:
        del position
        if not isinstance(feedback_memory, MemoryAttentionState):
            raise TypeError("multiscale Memory Attention feedback requires MemoryAttentionState")
        if token is None or token.shape != (feedback_memory.batch_size, 1):
            raise ValueError("multiscale Memory Attention update requires token [B,1]")
        self._validate_input_ids(token)
        if new_hidden.shape != (
            feedback_memory.batch_size,
            1,
            feedback_memory.hidden_size,
        ):
            raise ValueError("new_hidden must be [B,1,D]")

        new_record = self.writer(new_hidden).detach()
        write_positions = feedback_memory.next_sequence_positions[:, None]
        next_positions = feedback_memory.next_sequence_positions + 1
        candidate_memories = torch.cat((feedback_memory.memories, new_record), dim=1)
        candidate_valid = torch.cat(
            (
                feedback_memory.valid,
                torch.ones(
                    (feedback_memory.batch_size, 1),
                    dtype=torch.bool,
                    device=feedback_memory.valid.device,
                ),
            ),
            dim=1,
        )
        candidate_positions = torch.cat(
            (feedback_memory.positions, write_positions), dim=1
        )
        selection, selected_valid = self._selection(
            candidate_positions,
            candidate_valid,
            next_positions,
        )
        memories = self._gather_rows(candidate_memories, selection)
        positions = torch.gather(candidate_positions, 1, selection)

        if feedback_memory.projected_keys is None:
            return self._project_state(
                memories,
                selected_valid,
                positions,
                next_positions,
            )

        assert feedback_memory.projected_values is not None
        if len(feedback_memory.projected_keys) != len(self.memory_readers):
            raise ValueError("MemoryAttentionState projected K/V does not match memory-attention readers")
        projected_keys: list[torch.Tensor] = []
        projected_values: list[torch.Tensor] = []
        for cache_index, reader in enumerate(self.memory_readers.values()):
            new_key, new_value = reader.project_memory(
                new_record,
                position_ids=write_positions,
            )
            candidate_key = torch.cat(
                (feedback_memory.projected_keys[cache_index], new_key), dim=2
            )
            candidate_value = torch.cat(
                (feedback_memory.projected_values[cache_index], new_value), dim=2
            )
            gather_index = selection[:, None, :, None].expand(
                -1, candidate_key.shape[1], -1, candidate_key.shape[-1]
            )
            projected_keys.append(
                torch.gather(candidate_key, 2, gather_index).detach()
            )
            projected_values.append(
                torch.gather(candidate_value, 2, gather_index).detach()
            )

        return MemoryAttentionState(
            memories=memories.detach(),
            valid=selected_valid.detach(),
            positions=positions.detach(),
            next_sequence_positions=next_positions.detach(),
            projected_keys=tuple(projected_keys),
            projected_values=tuple(projected_values),
        )


MultiscaleBankVariant = MultiscaleMemoryAttentionVariant

__all__ = ["MultiscaleMemoryAttentionVariant", "MultiscaleBankVariant"]
