# coding=utf-8
# Derived from the Apache-2.0-licensed Hugging Face Transformers Mistral
# implementation (v4.45.2), Copyright 2023 Mistral AI and the HuggingFace
# Inc. team. This local version intentionally removes framework machinery.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    flex_local_attention,
    local_window_attention,
    reference_attention,
    retained_multiresolution_indices,
)
from .config import MistralConfig

AttentionBackend = Literal["auto", "reference", "flex", "local"]


@dataclass
class LayerKVCache:
    key: torch.Tensor  # [B, Hkv, S, D], already RoPE-rotated
    value: torch.Tensor  # [B, Hkv, S, D]
    start_pos: int
    key_valid: torch.Tensor | None = None  # optional bool [B,S]
    positions: torch.Tensor | None = None  # optional explicit integer [B,S]
    next_pos: int | None = None

    def __post_init__(self) -> None:
        if self.key_valid is not None:
            if self.key_valid.dtype != torch.bool or self.key_valid.shape != (self.key.shape[0], self.key.shape[-2]):
                raise ValueError("LayerKVCache.key_valid must be bool [B,S]")
        if self.positions is not None:
            if self.positions.shape != (self.key.shape[0], self.key.shape[-2]) or (
                self.positions.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("LayerKVCache.positions must be integer [B,S]")
        if self.next_pos is not None and int(self.next_pos) < 0:
            raise ValueError("LayerKVCache.next_pos must be non-negative")

    @property
    def seq_len(self) -> int:
        return int(self.key.shape[-2])

    @property
    def next_position(self) -> int:
        if self.next_pos is not None:
            return int(self.next_pos)
        return self.start_pos + self.seq_len


@dataclass
class BaseModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: tuple[LayerKVCache, ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: tuple[LayerKVCache, ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MistralMLP(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class MistralAttention(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        layer_idx: int,
        *,
        attention_backend: AttentionBackend = "auto",
        compile_flex: bool = True,
        flex_block_size: int = 128,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_backend = attention_backend
        self.compile_flex = compile_flex
        self.flex_block_size = flex_block_size

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.head_dim)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.sparse_attention_stride: int | None = None
        self.sparse_attention_window = 0

    def configure_sparse_attention(
        self,
        *,
        stride: int | None,
        window: int,
    ) -> None:
        window = int(window)
        if window < 0:
            raise ValueError("sparse attention window must be non-negative")
        if window:
            if self.config.sliding_window is None:
                raise ValueError("Strided Attention requires a finite sliding_window")
            if stride is None or int(stride) <= 0:
                raise ValueError("sparse attention stride must be positive")
            self.sparse_attention_stride = int(stride)
        else:
            self.sparse_attention_stride = None
        self.sparse_attention_window = window

    def _resolve_backend(
        self,
        hidden_states: torch.Tensor,
        *,
        past_key_value: LayerKVCache | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        fast_attention_compatible: bool | None = None,
    ) -> Literal["reference", "flex", "local"]:
        backend = self.attention_backend
        if backend == "reference":
            return "reference"
        if backend not in {"auto", "flex", "local"}:
            raise ValueError(f"unknown attention backend: {backend}")

        # Optimized full-sequence paths are intentionally restricted to the
        # common unpadded, zero-based contiguous prefill/training case. Cached
        # decode has at most W visible keys, so the obvious reference path is
        # already O(W) per generated token and is easier to audit.
        if fast_attention_compatible is None:
            can_fast = (
                past_key_value is None
                and hidden_states.shape[1] == position_ids.shape[1]
            )
            if can_fast:
                expected = torch.arange(
                    hidden_states.shape[1], device=position_ids.device, dtype=position_ids.dtype
                )[None, :].expand_as(position_ids)
                can_fast = bool(torch.equal(position_ids, expected))
        else:
            can_fast = fast_attention_compatible

        # FlexAttention is used only when dropout is zero because this austere
        # wrapper intentionally avoids carrying an explicit score_mod/dropout
        # implementation. The local PyTorch backend supports ordinary dropout.
        can_flex = can_fast and self.attention_dropout == 0.0

        if backend == "flex":
            return "flex" if can_flex else "reference"
        if backend == "local":
            return "local" if can_fast else "reference"

        # `auto` is device-aware: compiled sparse FlexAttention on CUDA, exact
        # O(T*W) local-window tensor math on Apple MPS, and the explicit dense
        # reference on CPU. The CPU local backend remains directly selectable
        # for tests and benchmarks.
        if hidden_states.device.type == "cuda" and can_flex:
            return "flex"
        if hidden_states.device.type == "mps" and can_fast:
            return "local"
        return "reference"

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
        fast_attention_compatible: bool | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is None:
            all_key_states = key_states
            all_value_states = value_states
            key_positions = position_ids
        else:
            if past_key_value.key.shape[:2] != (bsz, self.num_key_value_heads):
                raise ValueError("past KV cache has incompatible batch/head dimensions")
            if not torch.equal(position_ids, position_ids[:1].expand_as(position_ids)):
                raise ValueError("cached decoding requires common absolute positions across the batch")
            all_key_states = torch.cat((past_key_value.key, key_states), dim=-2)
            all_value_states = torch.cat((past_key_value.value, value_states), dim=-2)
            if past_key_value.positions is None:
                old_positions = torch.arange(
                    past_key_value.start_pos,
                    past_key_value.start_pos + past_key_value.seq_len,
                    device=position_ids.device,
                    dtype=position_ids.dtype,
                )[None, :].expand(bsz, -1)
            else:
                old_positions = past_key_value.positions.to(
                    device=position_ids.device, dtype=position_ids.dtype
                )
            key_positions = torch.cat((old_positions, position_ids), dim=-1)

        # `attention_mask` marks whether the *current* sequence positions may
        # be exposed as self-attention K/V entries.  Cached validity is carried
        # explicitly so write-only control tokens remain hidden on later decode
        # steps without disturbing their absolute position or KV-cache length.
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != bsz:
                raise ValueError("attention_mask must have shape [B, T]")
            if attention_mask.shape[1] < q_len:
                raise ValueError("attention_mask is shorter than current query sequence")
            current_valid = attention_mask[:, -q_len:].to(torch.bool)
        else:
            current_valid = torch.ones((bsz, q_len), dtype=torch.bool, device=hidden_states.device)
        if past_key_value is None:
            key_padding_mask = current_valid
        else:
            if past_key_value.key_valid is None:
                past_valid = torch.ones(
                    (bsz, past_key_value.seq_len), dtype=torch.bool, device=hidden_states.device
                )
            else:
                past_valid = past_key_value.key_valid.to(device=hidden_states.device)
            key_padding_mask = torch.cat((past_valid, current_valid), dim=1)

        backend = self._resolve_backend(
            hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            fast_attention_compatible=fast_attention_compatible,
        )
        if backend == "flex":
            attn_output = flex_local_attention(
                query_states,
                all_key_states,
                all_value_states,
                sliding_window=self.config.sliding_window,
                key_padding_mask=key_padding_mask,
                compile_kernel=self.compile_flex,
                block_size=self.flex_block_size,
                sparse_stride=self.sparse_attention_stride,
                sparse_window=self.sparse_attention_window,
            )
        elif backend == "local":
            attn_output = local_window_attention(
                query_states,
                all_key_states,
                all_value_states,
                sliding_window=self.config.sliding_window,
                key_padding_mask=key_padding_mask,
                dropout_p=self.attention_dropout,
                training=self.training,
                sparse_stride=self.sparse_attention_stride,
                sparse_window=self.sparse_attention_window,
            )
        else:
            attn_output = reference_attention(
                query_states,
                all_key_states,
                all_value_states,
                query_positions=position_ids,
                key_positions=key_positions,
                sliding_window=self.config.sliding_window,
                key_padding_mask=key_padding_mask,
                dropout_p=self.attention_dropout,
                training=self.training,
                sparse_stride=self.sparse_attention_stride,
                sparse_window=self.sparse_attention_window,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        new_cache = None
        if use_cache:
            next_position = int(position_ids[0, -1].item()) + 1
            if self.sparse_attention_window:
                assert self.sparse_attention_stride is not None
                recent_cache = max(int(self.config.sliding_window) - 1, 0)
                selection, selection_valid = retained_multiresolution_indices(
                    key_positions,
                    key_padding_mask,
                    torch.full(
                        (bsz,),
                        next_position,
                        device=key_positions.device,
                        dtype=key_positions.dtype,
                    ),
                    recent_window=recent_cache,
                    sparse_stride=self.sparse_attention_stride,
                    sparse_window=self.sparse_attention_window,
                )
                gather_key = selection[:, None, :, None].expand(
                    -1, self.num_key_value_heads, -1, self.head_dim
                )
                cache_key = torch.gather(all_key_states, 2, gather_key).detach()
                cache_value = torch.gather(all_value_states, 2, gather_key).detach()
                cache_positions = torch.gather(key_positions, 1, selection).detach()
                gathered_valid = torch.gather(key_padding_mask, 1, selection)
                cache_valid = (selection_valid & gathered_valid).detach()
                if bool(selection_valid[0].any()):
                    first = int(selection_valid[0].long().argmax().item())
                    start_pos = int(cache_positions[0, first].item())
                else:
                    start_pos = next_position
                new_cache = LayerKVCache(
                    cache_key,
                    cache_value,
                    start_pos,
                    cache_valid,
                    cache_positions,
                    next_position,
                )
            else:
                keep = all_key_states.shape[-2]
                if self.config.sliding_window is not None:
                    # v4.45.2 full-mask semantics allow W total visible keys,
                    # including the current query. Therefore the next decode step
                    # needs at most W-1 cached previous keys.
                    keep = min(keep, max(self.config.sliding_window - 1, 0))
                if keep:
                    cache_key = all_key_states[:, :, -keep:, :].detach()
                    cache_value = all_value_states[:, :, -keep:, :].detach()
                    start_pos = int(key_positions[0, -keep].item())
                else:
                    cache_key = all_key_states[:, :, :0, :].detach()
                    cache_value = all_value_states[:, :, :0, :].detach()
                    start_pos = next_position
                cache_valid = key_padding_mask[:, -keep:].detach() if keep else key_padding_mask[:, :0].detach()
                new_cache = LayerKVCache(cache_key, cache_value, start_pos, cache_valid)

        return attn_output, new_cache


class MistralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        layer_idx: int,
        *,
        attention_backend: AttentionBackend = "auto",
        compile_flex: bool = True,
        flex_block_size: int = 128,
    ):
        super().__init__()
        self.self_attn = MistralAttention(
            config,
            layer_idx,
            attention_backend=attention_backend,
            compile_flex=compile_flex,
            flex_block_size=flex_block_size,
        )
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        past_key_value: LayerKVCache | None = None,
        use_cache: bool = False,
        fast_attention_compatible: bool | None = None,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        x, new_cache = self.self_attn(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            fast_attention_compatible=fast_attention_compatible,
        )
        hidden_states = residual + x

        residual = hidden_states
        x = self.post_attention_layernorm(hidden_states)
        x = self.mlp(x)
        hidden_states = residual + x
        return hidden_states, new_cache


class MistralModel(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        *,
        attention_backend: AttentionBackend = "auto",
        compile_flex: bool = True,
        flex_block_size: int = 128,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [
                MistralDecoderLayer(
                    config,
                    layer_idx,
                    attention_backend=attention_backend,
                    compile_flex=compile_flex,
                    flex_block_size=flex_block_size,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Sequence[LayerKVCache] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool = False,
    ) -> BaseModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            if input_ids is None or input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [B, T]")
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [B, T, D]")

        bsz, seq_len, _ = inputs_embeds.shape
        fast_attention_compatible = (
            past_key_values is None
            and position_ids is None
        )
        if use_cache is None:
            use_cache = self.config.use_cache
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must have one cache per layer")

        cache_start: int | None = None
        if past_key_values:
            next_positions = {cache.next_position for cache in past_key_values}
            if len(next_positions) != 1:
                raise ValueError("layer caches disagree on next absolute position")
            cache_start = next(iter(next_positions))

        if position_ids is None:
            start = 0 if cache_start is None else cache_start
            position_ids = torch.arange(
                start, start + seq_len, device=inputs_embeds.device, dtype=torch.long
            )[None, :].expand(bsz, -1)
        else:
            if position_ids.shape != (bsz, seq_len):
                raise ValueError("position_ids must have shape [B, T]")
            if use_cache:
                if cache_start is None:
                    offsets = torch.arange(
                        seq_len, device=position_ids.device, dtype=position_ids.dtype
                    )
                    expected = position_ids[:1, :1] + offsets[None, :]
                else:
                    expected = torch.arange(
                        cache_start,
                        cache_start + seq_len,
                        device=position_ids.device,
                        dtype=position_ids.dtype,
                    )[None, :]
                expected = expected.expand(bsz, -1)
                if not torch.equal(position_ids, expected):
                    raise ValueError(
                        "caching requires contiguous absolute position_ids shared across the batch"
                    )

        hidden_states = inputs_embeds
        all_hidden_states: list[torch.Tensor] | None = [] if output_hidden_states else None
        new_caches: list[LayerKVCache] | None = [] if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            past = past_key_values[i] if past_key_values is not None else None
            hidden_states, cache = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past,
                use_cache=bool(use_cache),
                fast_attention_compatible=fast_attention_compatible,
            )
            if new_caches is not None:
                assert cache is not None
                new_caches.append(cache)

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=tuple(new_caches) if new_caches is not None else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )


class MistralForCausalLM(nn.Module):
    def __init__(
        self,
        config: MistralConfig,
        *,
        attention_backend: AttentionBackend = "auto",
        compile_flex: bool = True,
        flex_block_size: int = 128,
        initialize: bool = True,
    ):
        super().__init__()
        self.config = config
        self.model = MistralModel(
            config,
            attention_backend=attention_backend,
            compile_flex=compile_flex,
            flex_block_size=flex_block_size,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if initialize:
            self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_attention_backend(self, backend: AttentionBackend, *, compile_flex: bool | None = None) -> None:
        if backend not in {"auto", "reference", "flex", "local"}:
            raise ValueError(f"unknown attention backend: {backend}")
        for layer in self.model.layers:
            layer.self_attn.attention_backend = backend
            if compile_flex is not None:
                layer.self_attn.compile_flex = compile_flex

    def set_sparse_attention(
        self,
        *,
        stride: int,
        window: int,
        layers: str | Sequence[int] = "all",
    ) -> tuple[int, ...]:
        layer_count = len(self.model.layers)
        if layers == "all":
            selected = tuple(range(layer_count))
        elif isinstance(layers, Sequence) and not isinstance(layers, (str, bytes)):
            selected = tuple(sorted(int(index) for index in layers))
            if not selected or len(selected) != len(set(selected)):
                raise ValueError("sparse attention layers must be non-empty and unique")
            if selected[0] < 0 or selected[-1] >= layer_count:
                raise ValueError(f"sparse attention layers must lie in [0, {layer_count - 1}]")
        else:
            raise ValueError("sparse attention layers must be 'all' or a sequence")
        selected_set = set(selected)
        for index, layer in enumerate(self.model.layers):
            layer.self_attn.configure_sparse_attention(
                stride=stride if index in selected_set else None,
                window=window if index in selected_set else 0,
            )
        return selected

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Sequence[LayerKVCache] | None = None,
        use_cache: bool | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> CausalLMOutput:
        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        loss = None
        if labels is not None:
            if labels.ndim != 2 or labels.shape[:2] != logits.shape[:2]:
                raise ValueError("labels must have shape [B, T] matching logits")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be non-empty [B, T]")
        if input_ids.shape[0] != 1:
            raise ValueError("generate currently supports batch size 1")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None")
        if max_new_tokens == 0:
            return input_ids
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        result = input_ids
        backbone = self.model(result, use_cache=True)
        cache = backbone.past_key_values
        logits = self.lm_head(backbone.last_hidden_state[:, -1:, :]).float()[:, -1, :]

        for step in range(max_new_tokens):
            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                if top_k is not None:
                    k = min(top_k, scaled.shape[-1])
                    threshold = torch.topk(scaled, k, dim=-1).values[:, -1:]
                    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
                next_token = torch.multinomial(F.softmax(scaled, dim=-1), 1)
            result = torch.cat((result, next_token), dim=1)
            if eos is not None and bool(torch.all(next_token.squeeze(-1) == eos).item()):
                break
            if step == max_new_tokens - 1:
                break
            assert cache is not None
            # The cache stores absolute positions; only the most recent window
            # is retained. There is no need to pass the full historical mask for
            # unpadded single-prompt generation.
            out = self(next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
        return result

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
