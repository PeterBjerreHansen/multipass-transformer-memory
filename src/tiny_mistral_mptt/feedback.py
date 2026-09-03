from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MemoryAttentionState:
    """Fixed-capacity chronological Memory Attention records for inference.

    ``memories`` has shape ``[B,W,D]`` and ``valid`` has shape ``[B,W]``.
    Valid entries are kept left-aligned in chronological order.  The fixed
    capacity makes per-example token-triggered writes batchable even when
    examples have different write counts.
    """

    memories: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor
    next_sequence_positions: torch.Tensor
    projected_keys: tuple[torch.Tensor, ...] | None = None
    projected_values: tuple[torch.Tensor, ...] | None = None

    def __post_init__(self) -> None:
        if self.memories.ndim != 3:
            raise ValueError("MemoryAttentionState.memories must be [B,W,D]")
        if self.valid.ndim != 2 or self.valid.shape != self.memories.shape[:2]:
            raise ValueError("MemoryAttentionState.valid must be bool [B,W]")
        if self.valid.dtype != torch.bool:
            raise ValueError("MemoryAttentionState.valid must have bool dtype")
        if self.positions.shape != self.valid.shape or self.positions.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("MemoryAttentionState.positions must be integer [B,W]")
        if self.next_sequence_positions.shape != (self.memories.shape[0],) or (
            self.next_sequence_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "MemoryAttentionState.next_sequence_positions must be integer [B]"
            )
        if bool((self.positions[self.valid] < 0).any()):
            raise ValueError("valid MemoryAttentionState positions must be non-negative")
        if bool((self.next_sequence_positions < 0).any()):
            raise ValueError("next Memory Attention sequence positions must be non-negative")
        if self.memories.shape[1] < 1:
            raise ValueError("MemoryAttentionState capacity must be positive")
        if (self.projected_keys is None) != (self.projected_values is None):
            raise ValueError("projected_keys and projected_values must be provided together")
        if self.projected_keys is not None:
            assert self.projected_values is not None
            if len(self.projected_keys) != len(self.projected_values):
                raise ValueError("projected K/V tuple lengths differ")
            for key, value in zip(self.projected_keys, self.projected_values, strict=True):
                if key.ndim != 4 or key.shape != value.shape:
                    raise ValueError("projected K/V must have matching [B,Hkv,W,Dh] shapes")
                if key.shape[0] != self.batch_size or key.shape[2] != self.capacity:
                    raise ValueError("projected K/V batch/capacity mismatch")

    @property
    def batch_size(self) -> int:
        return int(self.memories.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.memories.shape[-1])

    @property
    def capacity(self) -> int:
        return int(self.memories.shape[1])


@dataclass(frozen=True)
class HybridPassSource:
    """Full-stream sources produced by one Memory Attention/recurrent pass."""

    recurrent_hidden: torch.Tensor
    memory_attention_hidden: torch.Tensor

    def __post_init__(self) -> None:
        if self.recurrent_hidden.ndim != 3 or self.memory_attention_hidden.ndim != 3:
            raise ValueError("hybrid pass sources must be [B,T,D]")
        if self.recurrent_hidden.shape != self.memory_attention_hidden.shape:
            raise ValueError("hybrid recurrent/attention pass sources must have equal shapes")


@dataclass(frozen=True)
class HybridFeedbackState:
    """One preceding-ordinary-token emitted record plus retained attention records."""

    recurrent_memory: torch.Tensor
    memory_attention: MemoryAttentionState

    def __post_init__(self) -> None:
        if self.recurrent_memory.ndim != 3 or self.recurrent_memory.shape[1] != 1:
            raise ValueError("HybridFeedbackState.recurrent_memory must be [B,1,D]")
        if self.recurrent_memory.shape[0] != self.memory_attention.batch_size:
            raise ValueError("hybrid recurrent/attention batch sizes differ")
        if self.recurrent_memory.shape[-1] != self.memory_attention.hidden_size:
            raise ValueError("hybrid recurrent/attention hidden dimensions differ")

    @property
    def batch_size(self) -> int:
        return int(self.recurrent_memory.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.recurrent_memory.shape[-1])


FeedbackMemory = torch.Tensor | MemoryAttentionState | HybridFeedbackState


def feedback_shape(memory: FeedbackMemory) -> tuple[int, int]:
    """Return ``(batch_size, hidden_size)`` for any feedback-state type."""
    if isinstance(memory, torch.Tensor):
        if memory.ndim != 3:
            raise ValueError("tensor feedback memory must be [B,M,D]")
        return int(memory.shape[0]), int(memory.shape[-1])
    return memory.batch_size, memory.hidden_size
