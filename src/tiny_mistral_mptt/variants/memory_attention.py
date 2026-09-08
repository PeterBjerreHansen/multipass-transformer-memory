from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from tiny_mistral.attention.multiresolution import retained_multiresolution_indices
from tiny_mistral.modeling import (
    LayerKVCache,
    MistralForCausalLM,
    MistralRMSNorm,
    MistralRotaryEmbedding,
    rotate_half,
)

from ..attention.memory_local import (
    memory_attention,
    strict_past_local_attention,
    strict_past_memory_attention,
    strict_past_dense_and_strided_memory_attention,
)
from ..feedback import MemoryAttentionState
from ..config import canonical_memory_write_mode
from .multipass import MultiPassVariant
from .memory_modules import (
    MemoryWriter as MemoryAttentionWriter,
)
from .memory_attention_fusion import (
    MEMORY_ATTENTION_FUSIONS,
    MemoryAttentionFusion,
)
from .decoder import DecoderRun as MemoryAttentionCoreRun, run_memory_decoder


MEMORY_WRITE_MODES = {"dense", "strided", "periodic"}
MEMORY_POSITION_ENCODINGS = {"rope", "none"}
MEMORY_READER_INITIALIZATIONS = {"zero_output", "aligned_gqa"}


class MemoryAttentionReader(nn.Module):
    """GQA reader over previous-pass memory records."""

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        window: int,
        num_key_value_heads: int | None = None,
        position_encoding: str = "rope",
        initialization: str = "zero_output",
        initialization_seed: int,
    ):
        super().__init__()
        config = backbone.config
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(
            config.num_key_value_heads
            if num_key_value_heads is None
            else num_key_value_heads
        )
        if (
            self.num_key_value_heads < 1
            or self.num_key_value_heads > self.num_heads
            or self.num_heads % self.num_key_value_heads != 0
        ):
            raise ValueError(
                "memory_num_key_value_heads must be a positive divisor of num_attention_heads"
            )
        self.head_dim = int(config.head_dim)
        self.window = int(window)
        self.dropout_p = float(config.attention_dropout)
        if position_encoding not in MEMORY_POSITION_ENCODINGS:
            raise ValueError("position_encoding must be 'rope' or 'none'")
        self.position_encoding = str(position_encoding)
        if initialization not in MEMORY_READER_INITIALIZATIONS:
            raise ValueError(
                "initialization must be 'zero_output' or 'aligned_gqa'"
            )
        self.initialization = str(initialization)
        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=int(config.max_position_embeddings),
            base=float(config.rope_theta),
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.query_norm = MistralRMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
            self.memory_norm = MistralRMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=False
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
            )
            self.o_proj = nn.Linear(
                self.num_heads * self.head_dim, self.hidden_size, bias=False
            )
            if self.initialization == "zero_output":
                std = float(config.initializer_range)
                for module in (self.q_proj, self.k_proj, self.v_proj):
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                # Retrofitting a pretrained backbone starts as an exact no-op.
                # The output projection learns first; once it moves away from
                # zero, gradients reach Q/K/V and the writer.
                nn.init.zeros_(self.o_proj.weight)
            else:
                self._initialize_aligned_gqa()

    def _initialize_aligned_gqa(self) -> None:
        """Initialize a norm-calibrated content reader in backbone coordinates.

        Each KV head pools the matching query-head chunks. Q uses the same
        pooled coordinates, while O maps each repeated GQA head back with the
        reciprocal scale. This is an orthogonal projection onto the subspace
        shared by each GQA group, rather than a literal identity (which cannot
        exist when there are fewer KV than query heads).
        """
        projected_width = self.num_heads * self.head_dim
        if projected_width != self.hidden_size:
            raise ValueError(
                "aligned_gqa requires hidden_size == num_attention_heads * head_dim"
            )
        group_size = self.num_heads // self.num_key_value_heads
        scale = group_size ** -0.5
        with torch.no_grad():
            self.q_proj.weight.zero_()
            self.k_proj.weight.zero_()
            self.v_proj.weight.zero_()
            self.o_proj.weight.zero_()
            for kv_head in range(self.num_key_value_heads):
                query_start = kv_head * group_size
                for offset in range(self.head_dim):
                    kv_row = kv_head * self.head_dim + offset
                    for group_offset in range(group_size):
                        query_head = query_start + group_offset
                        feature = query_head * self.head_dim + offset
                        query_row = feature
                        self.k_proj.weight[kv_row, feature] = scale
                        self.v_proj.weight[kv_row, feature] = scale
                        self.o_proj.weight[feature, query_row] = scale
                        for source_offset in range(group_size):
                            source_head = query_start + source_offset
                            source_feature = source_head * self.head_dim + offset
                            self.q_proj.weight[query_row, source_feature] = scale

    @staticmethod

    def _validate_positions(
        position_ids: torch.Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
        label: str,
    ) -> torch.Tensor:
        if position_ids is None:
            raise ValueError(f"{label} are required when memory RoPE is enabled")
        if position_ids.shape != (batch_size, sequence_length) or position_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(f"{label} must be integer [B,T]")
        if bool((position_ids < 0).any()):
            raise ValueError(f"{label} must be non-negative")
        return position_ids

    def _apply_position_encoding(
        self,
        states: torch.Tensor,
        position_ids: torch.Tensor | None,
        *,
        label: str,
    ) -> torch.Tensor:
        if self.position_encoding == "none":
            return states
        positions = self._validate_positions(
            position_ids,
            batch_size=states.shape[0],
            sequence_length=states.shape[-2],
            label=label,
        ).to(device=states.device)
        cos, sin = self.rotary_emb(states, positions)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        return (states * cos) + (rotate_half(states) * sin)

    def project_query(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must be [B,T,D] with the reader hidden size")
        bsz, query_len, _ = hidden_states.shape
        query = self.q_proj(self.query_norm(hidden_states))
        query = query.view(bsz, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        return self._apply_position_encoding(
            query, position_ids, label="query position_ids"
        )

    def project_memory(
        self,
        memory_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if memory_states.ndim != 3 or memory_states.shape[-1] != self.hidden_size:
            raise ValueError("memory_states must be [B,M,D] with the reader hidden size")
        bsz, memory_len, _ = memory_states.shape
        memory = self.memory_norm(memory_states)
        key = self.k_proj(memory).view(
            bsz, memory_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(memory).view(
            bsz, memory_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        key = self._apply_position_encoding(
            key, position_ids, label="memory position_ids"
        )
        return key, value

    def _project(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        query_position_ids: torch.Tensor | None = None,
        memory_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3 or memory_states.ndim != 3:
            raise ValueError("hidden_states and memory_states must be [B,T,D]")
        if hidden_states.shape[0] != memory_states.shape[0]:
            raise ValueError("hidden and memory batch sizes differ")
        return (
            self.project_query(hidden_states, position_ids=query_position_ids),
            *self.project_memory(memory_states, position_ids=memory_position_ids),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        query_position_ids: torch.Tensor | None = None,
        memory_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape != memory_states.shape:
            raise ValueError("hidden_states and memory_states must share [B,T,D]")
        bsz, seq_len, _ = hidden_states.shape
        query, key, value = self._project(
            hidden_states,
            memory_states,
            query_position_ids=query_position_ids,
            memory_position_ids=memory_position_ids,
        )
        output = strict_past_local_attention(
            query,
            key,
            value,
            window=self.window,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(output)

    def forward_memory(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
        memory_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend to memory records whose entries are already strictly in the past."""
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("cached Memory Attention query must be [B,1,D]")
        if memory_states.shape[1] > self.window:
            raise ValueError("cached memory records exceed configured window")
        key, value = self.project_memory(
            memory_states, position_ids=memory_position_ids
        )
        return self.forward_projected_memory(
            hidden_states,
            key,
            value,
            memory_mask=memory_mask,
            query_position_ids=query_position_ids,
        )

    def forward_projected_memory(
        self,
        hidden_states: torch.Tensor,
        projected_keys: torch.Tensor,
        projected_values: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read memory records whose K/V projections were computed at write time."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must be [B,T,D] with the reader hidden size")
        if hidden_states.shape[1] != 1:
            raise ValueError("cached Memory Attention query must be [B,1,D]")
        if projected_keys.ndim != 4 or projected_keys.shape != projected_values.shape:
            raise ValueError("projected K/V must have matching [B,Hkv,M,Dh] shapes")
        if projected_keys.shape[0] != hidden_states.shape[0]:
            raise ValueError("hidden and projected memory batch sizes differ")
        if projected_keys.shape[1] != self.num_key_value_heads:
            raise ValueError("projected memory has an incompatible KV-head count")
        if projected_keys.shape[-1] != self.head_dim:
            raise ValueError("projected memory has an incompatible head dimension")
        if projected_keys.shape[2] > self.window:
            raise ValueError("cached memory records exceed configured window")
        query = self.project_query(
            hidden_states, position_ids=query_position_ids
        )
        output = memory_attention(
            query,
            projected_keys,
            projected_values,
            memory_mask=memory_mask,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        bsz, query_len, _ = hidden_states.shape
        output = output.transpose(1, 2).contiguous().view(bsz, query_len, -1)
        return self.o_proj(output)

    def forward_full_memory(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        writes_before: torch.Tensor,
        memory_mask: torch.Tensor,
        query_position_ids: torch.Tensor | None = None,
        memory_position_ids: torch.Tensor | None = None,
        dense: bool = False,
    ) -> torch.Tensor:
        """Attend to the last ``window`` committed records at each query token."""
        if hidden_states.ndim != 3 or memory_states.ndim != 3:
            raise ValueError("hidden_states and memory_states must be [B,T,D]")
        bsz, query_len, _ = hidden_states.shape
        query, key, value = self._project(
            hidden_states,
            memory_states,
            query_position_ids=query_position_ids,
            memory_position_ids=memory_position_ids,
        )
        output = strict_past_memory_attention(
            query,
            key,
            value,
            writes_before=writes_before,
            memory_mask=memory_mask,
            window=self.window,
            dropout_p=self.dropout_p,
            training=self.training,
            dense=dense,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, query_len, -1)
        return self.o_proj(output)

    def forward_dense_and_strided_memory(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        memory_mask: torch.Tensor,
        dense_window: int,
        sparse_stride: int,
        sparse_window: int,
        query_position_ids: torch.Tensor,
        memory_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Attend once over the dense-recent/sparse-old memory union."""
        if hidden_states.ndim != 3 or memory_states.ndim != 3:
            raise ValueError("hidden_states and memory_states must be [B,T,D]")
        bsz, query_len, _ = hidden_states.shape
        query, key, value = self._project(
            hidden_states,
            memory_states,
            query_position_ids=query_position_ids,
            memory_position_ids=memory_position_ids,
        )
        output = strict_past_dense_and_strided_memory_attention(
            query,
            key,
            value,
            query_positions=query_position_ids,
            memory_positions=memory_position_ids,
            memory_mask=memory_mask,
            dense_window=dense_window,
            sparse_stride=sparse_stride,
            sparse_window=sparse_window,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, query_len, -1)
        return self.o_proj(output)


@dataclass(frozen=True)
class MemoryAttentionBatch:
    """Compact full-sequence memory records with original sequence coordinates."""

    memories: torch.Tensor  # [B,M,D], padded chronologically per example
    valid: torch.Tensor  # bool [B,M]
    writes_before: torch.Tensor  # [B,T]
    memory_positions: torch.Tensor  # integer [B,M], original sequence positions
    query_positions: torch.Tensor  # integer [B,T], original sequence positions

    def __post_init__(self) -> None:
        if self.memories.ndim != 3:
            raise ValueError("MemoryAttentionBatch.memories must be [B,M,D]")
        if self.valid.shape != self.memories.shape[:2] or self.valid.dtype != torch.bool:
            raise ValueError("MemoryAttentionBatch.valid must be bool [B,M]")
        if self.writes_before.ndim != 2:
            raise ValueError("MemoryAttentionBatch.writes_before must be [B,T]")
        if self.writes_before.shape[0] != self.memories.shape[0]:
            raise ValueError("Memory Attention batch sizes differ")
        if self.memory_positions.shape != self.valid.shape or (
            self.memory_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("MemoryAttentionBatch.memory_positions must be integer [B,M]")
        if self.query_positions.shape != self.writes_before.shape or (
            self.query_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("MemoryAttentionBatch.query_positions must be integer [B,T]")
        if bool((self.memory_positions[self.valid] < 0).any()):
            raise ValueError("valid MemoryAttentionBatch memory positions must be non-negative")
        if bool((self.query_positions < 0).any()):
            raise ValueError("MemoryAttentionBatch query positions must be non-negative")


class MemoryAttentionVariant(MultiPassVariant):
    """Memory Attention with dense, strided, or explicit-memory-token writes.

    Dense mode writes every top state; strided mode writes selected ordinary-token top states. Memory-token mode
    treats ID ``backbone.config.vocab_size`` as an input-only ``<MEM>`` control
    position with its own learned embedding; that ID is never an LM output
    class. A MEM state predicts nothing and writes exactly one memory record.

    """

    variant_name = "memory_attention"
    supports_cached_feedback = True

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        memory_window: int = 32,
        memory_pattern: str | None = None,
        memory_dense_window: int = 32,
        memory_sparse_window: int = 32,
        memory_sparse_stride: int = 32,
        memory_write_mode: str | None = None,
        memory_write_stride: int = 8,
        memory_layers: str | list[int] = "all",
        memory_position_encoding: str = "rope",
        memory_num_key_value_heads: int | None = None,
        memory_reader_initialization: str = "zero_output",
        memory_attention_fusion: str = "residual",
        memory_attention_controller_hidden_size: int | None = None,
        initialization_seed: int = 4242,
    ):
        super().__init__(backbone)
        if memory_write_mode is None:
            memory_write_mode = "dense" if memory_pattern in {"dense", "dense_and_strided"} else "strided"
        if memory_pattern is None:
            memory_pattern = "strided" if canonical_memory_write_mode(memory_write_mode) == "periodic" else "dense"
        if memory_pattern not in {"dense", "strided", "dense_and_strided"}:
            raise ValueError("memory_pattern must be dense, strided, or dense_and_strided")
        if memory_pattern == "dense_and_strided":
            if memory_write_mode != "dense":
                raise ValueError("dense-and-strided retention requires a dense source")
            if memory_dense_window < 0 or memory_sparse_window < 0 or memory_dense_window + memory_sparse_window <= 0:
                raise ValueError("dense-and-strided windows must be non-negative with non-zero total")
            if memory_sparse_stride <= 0:
                raise ValueError("memory_sparse_stride must be positive")
            memory_window = memory_dense_window + memory_sparse_window
            memory_write_stride = 1
        elif memory_pattern == "strided" and canonical_memory_write_mode(memory_write_mode) != "periodic":
            raise ValueError("strided memory_pattern conflicts with memory_write_mode")
        elif memory_pattern == "dense" and memory_write_mode != "dense":
            raise ValueError("dense memory_pattern conflicts with memory_write_mode")
        self.memory_pattern = memory_pattern
        self.memory_dense_window = int(memory_dense_window)
        self.memory_sparse_window = int(memory_sparse_window)
        self.memory_sparse_stride = int(memory_sparse_stride)
        if memory_window <= 0:
            raise ValueError("memory_window must be positive")
        memory_write_mode = canonical_memory_write_mode(memory_write_mode)
        if memory_write_mode not in MEMORY_WRITE_MODES:
            raise ValueError("memory_write_mode must be 'dense' or 'strided'")
        if memory_write_stride <= 0:
            raise ValueError("memory_write_stride must be positive")
        if memory_position_encoding not in MEMORY_POSITION_ENCODINGS:
            raise ValueError("memory_position_encoding must be 'rope' or 'none'")
        if memory_reader_initialization not in MEMORY_READER_INITIALIZATIONS:
            raise ValueError(
                "memory_reader_initialization must be zero_output or aligned_gqa"
            )
        if memory_attention_fusion not in MEMORY_ATTENTION_FUSIONS:
            raise ValueError(
                "memory_attention_fusion must be residual, destination_gated, "
                "attention_gated, or dual_gated"
            )
        if memory_attention_fusion == "residual":
            if memory_attention_controller_hidden_size is not None:
                raise ValueError(
                    "memory_attention_controller_hidden_size is not used by "
                    "residual fusion"
                )
        elif (
            memory_attention_controller_hidden_size is None
            or int(memory_attention_controller_hidden_size) < 1
        ):
            raise ValueError(
                "gated memory attention fusion requires a positive "
                "memory_attention_controller_hidden_size"
            )

        layer_count = len(backbone.model.layers)
        if memory_layers == "all":
            selected_layers = tuple(range(layer_count))
        elif isinstance(memory_layers, (list, tuple)) and memory_layers:
            selected_layers = tuple(sorted(int(layer) for layer in memory_layers))
            if len(selected_layers) != len(set(selected_layers)):
                raise ValueError("memory_layers indices must be unique")
            if selected_layers[0] < 0 or selected_layers[-1] >= layer_count:
                raise ValueError(
                    f"memory_layers must lie in [0, {layer_count - 1}]"
                )
        else:
            raise ValueError("memory_layers must be 'all' or a non-empty list")

        base_vocab = int(backbone.config.vocab_size)
        self.memory_window = int(memory_window)
        self.memory_write_mode = str(memory_write_mode)
        self.memory_write_stride = int(memory_write_stride)
        self.memory_layers = selected_layers
        self.memory_position_encoding = str(memory_position_encoding)
        self.memory_reader_initialization = str(memory_reader_initialization)
        self.memory_attention_fusion = str(memory_attention_fusion)
        self.memory_attention_controller_hidden_size = (
            None
            if memory_attention_controller_hidden_size is None
            else int(memory_attention_controller_hidden_size)
        )
        self.memory_num_key_value_heads = (
            int(backbone.config.num_key_value_heads)
            if memory_num_key_value_heads is None
            else int(memory_num_key_value_heads)
        )
        self.base_vocab_size = base_vocab

        hidden_size = int(backbone.config.hidden_size)
        self.writer = MemoryAttentionWriter(hidden_size)
        self.memory_readers = nn.ModuleDict(
            {
                str(layer_index): MemoryAttentionReader(
                    backbone,
                    window=self.memory_window,
                    num_key_value_heads=self.memory_num_key_value_heads,
                    position_encoding=self.memory_position_encoding,
                    initialization=self.memory_reader_initialization,
                    initialization_seed=int(initialization_seed) + layer_index,
                )
                for layer_index in self.memory_layers
            }
        )
        self.memory_attention_fusions = nn.ModuleDict(
            {
                str(layer_index): MemoryAttentionFusion(
                    hidden_size,
                    mode=self.memory_attention_fusion,
                    controller_hidden_size=(
                        self.memory_attention_controller_hidden_size
                    ),
                    initialization_seed=int(initialization_seed) + 10_000 + layer_index,
                )
                for layer_index in self.memory_layers
            }
        )
        self._reader_cache_index = {
            layer_index: cache_index
            for cache_index, layer_index in enumerate(self.memory_layers)
        }

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().added_parameters()
        yield from self.writer.parameters()
        yield from self.memory_readers.parameters()
        yield from self.memory_attention_fusions.parameters()

    def _fuse_memory(
        self,
        layer_index: int,
        destination: torch.Tensor,
        memory_delta: torch.Tensor,
        *,
        available: torch.Tensor,
    ) -> torch.Tensor:
        return self.memory_attention_fusions[str(layer_index)](
            destination, memory_delta, available=available
        )

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        # Avoid a host synchronization in every cached CUDA/MPS token step. The
        # checked data/config pipeline establishes the range before device
        # execution; CPU/direct-call paths retain eager, friendly validation.
        if input_ids.device.type == "cpu":
            if bool((input_ids < 0).any()):
                raise ValueError("input IDs must be non-negative")
            upper = self.base_vocab_size
            if bool((input_ids >= upper).any()):
                raise ValueError("input ID lies outside this variant's input vocabulary")

    def input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        return self.backbone.model.embed_tokens(input_ids)

    def write_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        if self.memory_write_mode == "dense":
            return torch.ones_like(input_ids, dtype=torch.bool)
        if self.memory_write_mode == "periodic":
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            row = (positions + 1).remainder(self.memory_write_stride).eq(0)
            return row[None, :].expand(input_ids.shape[0], -1)
        raise RuntimeError("invalid memory write mode")

    def sequence_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return physical input positions for strict-past cross-attention."""
        self._validate_input_ids(input_ids)
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device, dtype=torch.long
        )
        return positions[None, :].expand(input_ids.shape[0], -1)

    def next_sequence_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        return torch.full(
            (input_ids.shape[0],),
            input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        )

    def _compact_written_states(
        self,
        written_states: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact selected states chronologically, padding only across batch."""
        if written_states.ndim != 3 or write_mask.shape != written_states.shape[:2]:
            raise ValueError("written_states/write_mask shapes are incompatible")
        counts = write_mask.sum(dim=1)
        max_count = int(counts.max().item()) if counts.numel() else 0
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for batch_index in range(written_states.shape[0]):
            selected = written_states[batch_index, write_mask[batch_index], :]
            count = selected.shape[0]
            if count < max_count:
                padding = selected.new_zeros((max_count - count, selected.shape[-1]))
                selected = torch.cat((selected, padding), dim=0)
            rows.append(selected)
            masks.append(torch.arange(max_count, device=write_mask.device) < count)
        if not rows:
            return (
                written_states.new_zeros((0, 0, written_states.shape[-1])),
                write_mask.new_zeros((0, 0)),
            )
        return torch.stack(rows, dim=0), torch.stack(masks, dim=0)

    @staticmethod

    def _compact_written_positions(
        positions: torch.Tensor,
        write_mask: torch.Tensor,
        *,
        width: int,
    ) -> torch.Tensor:
        if positions.shape != write_mask.shape:
            raise ValueError("positions/write_mask shapes are incompatible")
        rows: list[torch.Tensor] = []
        for batch_index in range(positions.shape[0]):
            selected = positions[batch_index, write_mask[batch_index]]
            if selected.shape[0] < width:
                selected = torch.cat(
                    (
                        selected,
                        torch.zeros(
                            width - selected.shape[0],
                            device=positions.device,
                            dtype=positions.dtype,
                        ),
                    )
                )
            rows.append(selected)
        if not rows:
            return positions.new_zeros((0, width))
        return torch.stack(rows, dim=0)

    def build_memory(
        self,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> MemoryAttentionBatch:
        if previous_hidden.ndim != 3 or input_ids.ndim != 2:
            raise ValueError("previous_hidden must be [B,T,D] and input_ids [B,T]")
        if previous_hidden.shape[:2] != input_ids.shape:
            raise ValueError("previous_hidden and input_ids token shapes differ")
        mask = self.write_mask(input_ids)
        selected, valid = self._compact_written_states(previous_hidden, mask)
        query_positions = self.sequence_positions(input_ids)
        memory_positions = self._compact_written_positions(
            query_positions, mask, width=selected.shape[1]
        )
        memories = self.writer(selected)
        cumulative = mask.long().cumsum(dim=1)
        writes_before = cumulative - mask.long()
        return MemoryAttentionBatch(
            memories=memories,
            valid=valid,
            writes_before=writes_before,
            memory_positions=memory_positions,
            query_positions=query_positions,
        )

    @staticmethod

    def _cache_next_position(past_key_values: tuple[LayerKVCache, ...]) -> int:
        if not past_key_values:
            raise ValueError("past_key_values must not be empty")
        positions = {cache.next_position for cache in past_key_values}
        if len(positions) != 1:
            raise ValueError("layer caches disagree on next absolute position")
        return next(iter(positions))

    def _run_attention_core(
        self,
        token_embeddings: torch.Tensor,
        memory: MemoryAttentionBatch | MemoryAttentionState | None,
        *,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        query_position_ids: torch.Tensor | None = None,
        after_memory_attention: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
    ) -> MemoryAttentionCoreRun:
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must be [B,T,D]")
        if memory is not None and token_embeddings.shape[0] != memory.memories.shape[0]:
            raise ValueError("token and memory-state batch sizes differ")
        if memory is not None and token_embeddings.shape[-1] != memory.memories.shape[-1]:
            raise ValueError("token and memory-state hidden dimensions differ")
        cached_memory = isinstance(memory, MemoryAttentionState)
        if cached_memory and token_embeddings.shape[1] != 1:
            raise ValueError("cached Memory Attention query must contain exactly one token")
        if cached_memory and memory.capacity != self.memory_window:
            raise ValueError("cached Memory Attention capacity differs from memory_window")
        if past_key_values is not None and len(past_key_values) != len(self.backbone.model.layers):
            raise ValueError("past_key_values must contain one cache per layer")

        bsz, seq_len, _ = token_embeddings.shape
        if isinstance(memory, MemoryAttentionBatch):
            memory_query_positions = memory.query_positions
        elif isinstance(memory, MemoryAttentionState):
            if query_position_ids is None:
                query_position_ids = memory.next_sequence_positions[:, None]
            if query_position_ids.shape != (bsz, seq_len):
                raise ValueError("cached Memory Attention query positions must be [B,1]")
            memory_query_positions = query_position_ids
        else:
            memory_query_positions = None

        def read_memory(layer_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
            reader_key = str(layer_index)
            if memory is not None and reader_key in self.memory_readers:
                memory_reader = self.memory_readers[reader_key]
                if cached_memory:
                    assert isinstance(memory, MemoryAttentionState)
                    if memory.projected_keys is None:
                        memory_delta = memory_reader.forward_memory(
                            hidden_states,
                            memory.memories,
                            memory_mask=memory.valid,
                            query_position_ids=memory_query_positions,
                            memory_position_ids=memory.positions,
                        )
                    else:
                        if memory.projected_values is None or len(memory.projected_keys) != len(self.memory_readers):
                            raise ValueError("MemoryAttentionState projected K/V does not match Memory Attention readers")
                        cache_index = self._reader_cache_index[layer_index]
                        memory_delta = memory_reader.forward_projected_memory(
                            hidden_states,
                            memory.projected_keys[cache_index],
                            memory.projected_values[cache_index],
                            memory_mask=memory.valid,
                            query_position_ids=memory_query_positions,
                        )
                    available = memory.valid.any(dim=1, keepdim=True)
                else:
                    assert isinstance(memory, MemoryAttentionBatch)
                    memory_delta = self._full_memory_delta(
                        memory_reader,
                        hidden_states,
                        memory,
                        query_position_ids=memory_query_positions,
                    )
                    available = memory.writes_before.gt(0)
                hidden_states = self._fuse_memory(
                    layer_index,
                    hidden_states,
                    memory_delta,
                    available=available,
                )
            if after_memory_attention is not None:
                hidden_states = after_memory_attention(layer_index, hidden_states)
            return hidden_states

        return run_memory_decoder(
            self.backbone,
            token_embeddings,
            after_attention=read_memory,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def _full_memory_delta(
        self,
        memory_reader: MemoryAttentionReader,
        hidden_states: torch.Tensor,
        memory: MemoryAttentionBatch,
        *,
        query_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Read the selected strict-past pattern from a completed source pass."""
        if self.memory_pattern == "dense_and_strided":
            return memory_reader.forward_dense_and_strided_memory(
                hidden_states, memory.memories, memory_mask=memory.valid,
                dense_window=self.memory_dense_window,
                sparse_stride=self.memory_sparse_stride,
                sparse_window=self.memory_sparse_window,
                query_position_ids=query_position_ids,
                memory_position_ids=memory.memory_positions,
            )
        return memory_reader.forward_full_memory(
            hidden_states,
            memory.memories,
            writes_before=memory.writes_before,
            memory_mask=memory.valid,
            query_position_ids=query_position_ids,
            memory_position_ids=memory.memory_positions,
            dense=(
                self.memory_write_mode == "dense"
                or (
                    self.memory_write_mode == "periodic"
                    and self.memory_write_stride == 1
                )
            ),
        )

    def _run_attention_feedback_core(
        self,
        token_embeddings: torch.Tensor,
        memory: MemoryAttentionBatch | MemoryAttentionState,
        *,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        query_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...] | None]:
        """Compatibility wrapper for a memory-only decoder pass."""
        run = self._run_attention_core(
            token_embeddings,
            memory,
            past_key_values=past_key_values,
            use_cache=use_cache,
            query_position_ids=query_position_ids,
        )
        return run.hidden_states, run.past_key_values

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.build_memory(previous_hidden, input_ids)
        hidden, _ = self._run_attention_feedback_core(
            token_embeddings,
            memory,
            past_key_values=None,
            use_cache=False,
        )
        return hidden

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        memory = self.build_memory(previous_hidden, input_ids)
        hidden, cache = self._run_attention_feedback_core(
            token_embeddings,
            memory,
            past_key_values=None,
            use_cache=True,
        )
        if cache is None:
            raise RuntimeError("cached Memory Attention prefill did not return KV state")
        return hidden, cache

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: MemoryAttentionState,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if not isinstance(feedback_memory, MemoryAttentionState):
            raise TypeError("Memory Attention cached feedback requires MemoryAttentionState")
        if token is None:
            raise ValueError("cached Memory Attention requires the current token ID")
        query_positions = self._cached_query_positions(feedback_memory, token)
        hidden, cache = self._run_attention_feedback_core(
            token_embedding,
            feedback_memory,
            past_key_values=past_key_values,
            use_cache=True,
            query_position_ids=query_positions,
        )
        if cache is None:
            raise RuntimeError("cached Memory Attention token did not return KV state")
        return hidden, cache

    def _cached_query_positions(
        self, state: MemoryAttentionState, token: torch.Tensor
    ) -> torch.Tensor:
        if token.shape != (state.batch_size, 1):
            raise ValueError("cached Memory Attention token must be [B,1]")
        query_positions = state.next_sequence_positions[:, None]
        return query_positions

    def _state_from_memory_batch(self, memory: MemoryAttentionBatch) -> MemoryAttentionState:
        if self.memory_pattern == "dense_and_strided":
            return self._state_from_dense_and_strided_batch(memory)
        bsz, _, dim = memory.memories.shape
        result = memory.memories.new_zeros((bsz, self.memory_window, dim))
        valid = torch.zeros((bsz, self.memory_window), dtype=torch.bool, device=memory.memories.device)
        positions = torch.zeros(
            (bsz, self.memory_window), dtype=torch.long, device=memory.memories.device
        )
        for batch_index in range(bsz):
            row = memory.memories[batch_index, memory.valid[batch_index], :][-self.memory_window :]
            row_positions = memory.memory_positions[
                batch_index, memory.valid[batch_index]
            ][-self.memory_window :]
            count = row.shape[0]
            if count:
                result[batch_index, :count, :] = row
                valid[batch_index, :count] = True
                positions[batch_index, :count] = row_positions
        next_positions = memory.query_positions[:, -1] + 1
        return self._project_state(result, valid, positions, next_positions)

    def _project_state(
        self,
        memories: torch.Tensor,
        valid: torch.Tensor,
        positions: torch.Tensor,
        next_sequence_positions: torch.Tensor,
    ) -> MemoryAttentionState:
        projected_keys: list[torch.Tensor] = []
        projected_values: list[torch.Tensor] = []
        for reader in self.memory_readers.values():
            key, value = reader.project_memory(
                memories, position_ids=positions
            )
            projected_keys.append(key.detach())
            projected_values.append(value.detach())
        return MemoryAttentionState(
            memories=memories.detach(),
            valid=valid.detach(),
            positions=positions.detach(),
            next_sequence_positions=next_sequence_positions.detach(),
            projected_keys=tuple(projected_keys),
            projected_values=tuple(projected_values),
        )

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> MemoryAttentionState:
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError("hidden_states must be non-empty [B,T,D]")
        if input_ids is None:
            input_ids = torch.zeros(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)
        return self._state_from_memory_batch(self.build_memory(hidden_states, input_ids))

    def _write_trigger(
        self,
        *,
        token: torch.Tensor | None,
        position: int | None,
    ) -> torch.Tensor:
        if token is None or token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("cached write requires token [B,1]")
        if self.memory_write_mode == "dense":
            return torch.ones((token.shape[0],), dtype=torch.bool, device=token.device)
        if self.memory_write_mode == "periodic":
            if position is None:
                raise ValueError("strided cached write requires absolute position")
            trigger = (int(position) + 1) % self.memory_write_stride == 0
            return torch.full((token.shape[0],), trigger, dtype=torch.bool, device=token.device)
        raise RuntimeError("invalid memory write mode")

    def _append_memory(
        self,
        state: MemoryAttentionState,
        new_hidden: torch.Tensor,
        *,
        trigger: torch.Tensor,
        write_positions: torch.Tensor,
        next_sequence_positions: torch.Tensor,
    ) -> MemoryAttentionState:
        if new_hidden.ndim != 3 or new_hidden.shape[1] != 1:
            raise ValueError("new_hidden must be [B,1,D]")
        if trigger.shape != (new_hidden.shape[0],) or trigger.dtype != torch.bool:
            raise ValueError("trigger must be bool [B]")
        if write_positions.shape != trigger.shape or write_positions.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("write_positions must be integer [B]")
        if next_sequence_positions.shape != trigger.shape or (
            next_sequence_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("next_sequence_positions must be integer [B]")
        if state.batch_size != new_hidden.shape[0] or state.hidden_size != new_hidden.shape[-1]:
            raise ValueError("memory state and new hidden shapes are incompatible")
        new_record = self.writer(new_hidden).detach()
        memories = state.memories.clone()
        valid = state.valid.clone()
        positions = state.positions.clone()
        counts = valid.sum(dim=1, dtype=torch.long)
        full_trigger = trigger & counts.eq(self.memory_window)
        shifted_memories = torch.cat(
            (memories[:, 1:, :], torch.zeros_like(memories[:, :1, :])), dim=1
        )
        shifted_valid = torch.cat((valid[:, 1:], torch.zeros_like(valid[:, :1])), dim=1)
        shifted_positions = torch.cat(
            (positions[:, 1:], torch.zeros_like(positions[:, :1])), dim=1
        )
        memories = torch.where(full_trigger[:, None, None], shifted_memories, memories)
        valid = torch.where(full_trigger[:, None], shifted_valid, valid)
        positions = torch.where(full_trigger[:, None], shifted_positions, positions)
        write_index = counts.clamp(max=self.memory_window - 1)
        scatter_index = write_index[:, None, None].expand(-1, 1, new_record.shape[-1])
        candidate_memories = memories.scatter(1, scatter_index, new_record)
        candidate_valid = valid.scatter(1, write_index[:, None], torch.ones_like(trigger[:, None]))
        candidate_positions = positions.scatter(1, write_index[:, None], write_positions[:, None])
        memories = torch.where(trigger[:, None, None], candidate_memories, memories)
        valid = torch.where(trigger[:, None], candidate_valid, valid)
        positions = torch.where(trigger[:, None], candidate_positions, positions)
        if state.projected_keys is None:
            return self._project_state(
                memories, valid, positions, next_sequence_positions
            )

        assert state.projected_values is not None
        if len(state.projected_keys) != len(self.memory_readers):
            raise ValueError("MemoryAttentionState projected K/V does not match Memory Attention readers")
        projected_keys: list[torch.Tensor] = []
        projected_values: list[torch.Tensor] = []
        for cache_index, reader in enumerate(self.memory_readers.values()):
            new_key, new_value = reader.project_memory(
                new_record, position_ids=write_positions[:, None]
            )
            old_key = state.projected_keys[cache_index].clone()
            old_value = state.projected_values[cache_index].clone()
            shifted_key = torch.cat((old_key[:, :, 1:, :], torch.zeros_like(old_key[:, :, :1, :])), dim=2)
            shifted_value = torch.cat((old_value[:, :, 1:, :], torch.zeros_like(old_value[:, :, :1, :])), dim=2)
            old_key = torch.where(full_trigger[:, None, None, None], shifted_key, old_key)
            old_value = torch.where(full_trigger[:, None, None, None], shifted_value, old_value)
            scatter_index = write_index[:, None, None, None].expand(
                -1, old_key.shape[1], 1, old_key.shape[-1]
            )
            candidate_key = old_key.scatter(2, scatter_index, new_key)
            candidate_value = old_value.scatter(2, scatter_index, new_value)
            old_key = torch.where(trigger[:, None, None, None], candidate_key, old_key)
            old_value = torch.where(trigger[:, None, None, None], candidate_value, old_value)
            projected_keys.append(old_key.detach())
            projected_values.append(old_value.detach())
        return MemoryAttentionState(
            memories=memories.detach(),
            valid=valid.detach(),
            positions=positions.detach(),
            next_sequence_positions=next_sequence_positions.detach(),
            projected_keys=tuple(projected_keys),
            projected_values=tuple(projected_values),
        )

    def _append_feedback_memory(
        self,
        feedback_memory: MemoryAttentionState,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> MemoryAttentionState:
        if self.memory_pattern == "dense_and_strided":
            return self._append_dense_and_strided_memory(feedback_memory, new_hidden, token=token, position=position)
        if not isinstance(feedback_memory, MemoryAttentionState):
            raise TypeError("Memory Attention feedback requires MemoryAttentionState")
        trigger = self._write_trigger(token=token, position=position)
        if token is None:
            raise ValueError("Memory Attention feedback update requires current token")
        current_positions = feedback_memory.next_sequence_positions
        next_positions = current_positions + 1
        write_positions = current_positions
        return self._append_memory(
            feedback_memory,
            new_hidden,
            trigger=trigger,
            write_positions=write_positions,
            next_sequence_positions=next_positions,
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

    def _state_from_dense_and_strided_batch(self, memory: MemoryAttentionBatch) -> MemoryAttentionState:
        next_positions = memory.query_positions[:, -1] + 1
        selection, selected_valid = self._selection(
            memory.memory_positions,
            memory.valid,
            next_positions,
        )
        memories = self._gather_rows(memory.memories, selection)
        positions = torch.gather(memory.memory_positions, 1, selection)
        return self._project_state(
            memories,
            selected_valid,
            positions,
            next_positions,
        )

    def _append_dense_and_strided_memory(
        self,
        feedback_memory: MemoryAttentionState,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> MemoryAttentionState:
        del position
        if not isinstance(feedback_memory, MemoryAttentionState):
            raise TypeError("dense-and-strided Memory Attention feedback requires MemoryAttentionState")
        if token is None or token.shape != (feedback_memory.batch_size, 1):
            raise ValueError("dense-and-strided Memory Attention update requires token [B,1]")
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


__all__ = [
    "MemoryAttentionReader",
    "MemoryAttentionBatch",
    "MemoryAttentionWriter",
    "MemoryAttentionVariant",
    "MemoryAttentionCoreRun",
]
