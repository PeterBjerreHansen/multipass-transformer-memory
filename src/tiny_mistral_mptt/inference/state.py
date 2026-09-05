from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from tiny_mistral.modeling import LayerKVCache

from ..feedback import FeedbackMemory, feedback_shape


KVCache = tuple[LayerKVCache, ...]
DecodeMode = Literal["standard", "feedback"]


def cache_next_position(past_key_values: KVCache) -> int:
    """Return the common next absolute position encoded by a layer cache."""
    if not past_key_values:
        raise ValueError("past_key_values must contain one cache per decoder layer")
    positions = {cache.next_position for cache in past_key_values}
    if len(positions) != 1:
        raise ValueError("layer caches disagree on next absolute position")
    return next(iter(positions))


@dataclass(frozen=True)
class PassStreamState:
    """One causal stream retained by exact K-pass inference."""

    past_key_values: KVCache
    feedback_memory: FeedbackMemory
    last_hidden: torch.Tensor

    def __post_init__(self) -> None:
        if self.last_hidden.ndim != 3 or self.last_hidden.shape[1] != 1:
            raise ValueError("last_hidden must be [B,1,D]")
        batch_size, hidden_size = feedback_shape(self.feedback_memory)
        if batch_size != self.last_hidden.shape[0]:
            raise ValueError("feedback memory and last hidden batch sizes differ")
        if hidden_size != self.last_hidden.shape[-1]:
            raise ValueError("feedback memory and last hidden dimensions differ")
        cache_next_position(self.past_key_values)


@dataclass(frozen=True)
class ExactKPassState:
    """State for exact cached inference with ``K`` parallel pass streams."""

    prefill_passes: int
    streams: tuple[PassStreamState, ...]
    next_token_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.prefill_passes < 1:
            raise ValueError("prefill_passes must be positive")
        if len(self.streams) != self.prefill_passes:
            raise ValueError("exact state must contain exactly prefill_passes streams")
        if self.next_token_logits.ndim != 2:
            raise ValueError("next_token_logits must be [B,V]")
        batch_sizes = {stream.last_hidden.shape[0] for stream in self.streams}
        if batch_sizes != {self.next_token_logits.shape[0]}:
            raise ValueError("stream and logits batch sizes differ")
        positions = {cache_next_position(stream.past_key_values) for stream in self.streams}
        if len(positions) != 1:
            raise ValueError("pass streams disagree on next absolute position")

    @property
    def next_position(self) -> int:
        return cache_next_position(self.streams[-1].past_key_values)

    @property
    def last_hidden(self) -> torch.Tensor:
        return self.streams[-1].last_hidden


@dataclass(frozen=True)
class LiveFeedbackState:
    """Collapsed one-stream state after a K-pass prompt prefill.

    ``prefill_passes`` describes only how the prompt state was constructed.
    ``decode_mode`` independently selects ordinary cached decoding or the
    one-step feedback recurrence used for long continuations.
    """

    prefill_passes: int
    decode_mode: DecodeMode
    past_key_values: KVCache
    feedback_memory: FeedbackMemory | None
    last_hidden: torch.Tensor
    next_token_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.prefill_passes < 1:
            raise ValueError("prefill_passes must be positive")
        if self.decode_mode not in ("standard", "feedback"):
            raise ValueError(f"unknown decode mode {self.decode_mode!r}")
        if self.last_hidden.ndim != 3 or self.last_hidden.shape[1] != 1:
            raise ValueError("last_hidden must be [B,1,D]")
        if self.next_token_logits.ndim != 2:
            raise ValueError("next_token_logits must be [B,V]")
        if self.last_hidden.shape[0] != self.next_token_logits.shape[0]:
            raise ValueError("hidden and logits batch sizes differ")
        cache_next_position(self.past_key_values)
        if self.decode_mode == "standard":
            if self.feedback_memory is not None:
                raise ValueError("standard decoding must not carry feedback memory")
        else:
            if self.feedback_memory is None:
                raise ValueError("feedback decoding requires feedback memory")
            batch_size, hidden_size = feedback_shape(self.feedback_memory)
            if batch_size != self.last_hidden.shape[0]:
                raise ValueError("feedback memory and hidden batch sizes differ")
            if hidden_size != self.last_hidden.shape[-1]:
                raise ValueError("feedback memory and hidden dimensions differ")

    @property
    def feedback_enabled(self) -> bool:
        return self.decode_mode == "feedback"

    @property
    def next_position(self) -> int:
        return cache_next_position(self.past_key_values)
