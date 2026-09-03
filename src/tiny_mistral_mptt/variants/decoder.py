"""Shared research decoder execution without modifying the vendored backbone."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM


@dataclass(frozen=True)
class DecoderRun:
    hidden_states: torch.Tensor
    past_key_values: tuple[LayerKVCache, ...] | None


def run_memory_decoder(
    backbone: MistralForCausalLM,
    embeddings: torch.Tensor,
    *,
    after_attention: Callable[[int, torch.Tensor], torch.Tensor],
    past_key_values: tuple[LayerKVCache, ...] | None,
    use_cache: bool,
    attention_mask: torch.Tensor | None = None,
) -> DecoderRun:
    """Apply memory reads after self-attention and before each layer's MLP.

    Cached and full-sequence runs share the same ordering, absolute positions,
    and final normalization.
    """
    if embeddings.ndim != 3:
        raise ValueError("embeddings must be [B,T,D]")
    if past_key_values is not None and len(past_key_values) != len(backbone.model.layers):
        raise ValueError("past_key_values must contain one cache per layer")
    start = 0
    if past_key_values:
        positions = {cache.next_position for cache in past_key_values}
        if len(positions) != 1:
            raise ValueError("layer caches disagree on next position")
        start = next(iter(positions))
    batch, length, _ = embeddings.shape
    position_ids = torch.arange(
        start, start + length, device=embeddings.device, dtype=torch.long
    )[None, :].expand(batch, -1)
    hidden = embeddings
    caches = [] if use_cache else None
    for index, layer in enumerate(backbone.model.layers):
        attended, cache = layer.self_attn(
            layer.input_layernorm(hidden),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None if past_key_values is None else past_key_values[index],
            use_cache=use_cache,
            fast_attention_compatible=past_key_values is None,
        )
        hidden = after_attention(index, hidden + attended)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        if caches is not None:
            if cache is None:
                raise RuntimeError("cached memory decoder did not return KV state")
            caches.append(cache)
    return DecoderRun(
        backbone.model.norm(hidden),
        tuple(caches) if caches is not None else None,
    )
