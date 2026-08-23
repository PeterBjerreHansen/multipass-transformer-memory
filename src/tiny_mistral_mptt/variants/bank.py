from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from tiny_mistral.modeling import (
    LayerKVCache,
    MistralForCausalLM,
    MistralRMSNorm,
    MistralRotaryEmbedding,
    rotate_half,
)

from ..attention.memory_local import (
    memory_bank_attention,
    strict_past_local_attention,
    strict_past_bank_attention,
)
from ..feedback import BankState
from .multipass import MultiPassVariant


MEMORY_WRITE_MODES = {"dense", "periodic", "memory_token"}
MEMORY_TOKEN_VISIBILITIES = {"visible", "write_only"}
MEMORY_POSITION_ENCODINGS = {"rope", "none"}


class BankReader(nn.Module):
    """Mistral-shaped GQA cross-attention into a strict-past local memory bank."""

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        window: int,
        position_encoding: str = "rope",
        initialization_seed: int,
    ):
        super().__init__()
        config = backbone.config
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(config.head_dim)
        self.window = int(window)
        self.dropout_p = float(config.attention_dropout)
        if position_encoding not in MEMORY_POSITION_ENCODINGS:
            raise ValueError("position_encoding must be 'rope' or 'none'")
        self.position_encoding = str(position_encoding)
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
            std = float(config.initializer_range)
            for module in (self.q_proj, self.k_proj, self.v_proj):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            # Retrofitting a pretrained backbone must start as an exact no-op.
            # The output projection learns first; once it moves away from zero,
            # gradients reach Q/K/V and the writer on subsequent updates.
            nn.init.zeros_(self.o_proj.weight)

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

    def forward_bank(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
        memory_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend to a bank whose entries are already strictly in the past."""
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("cached Bank query must be [B,1,D]")
        if memory_states.shape[1] > self.window:
            raise ValueError("cached memory bank exceeds configured window")
        key, value = self.project_memory(
            memory_states, position_ids=memory_position_ids
        )
        return self.forward_projected_bank(
            hidden_states,
            key,
            value,
            memory_mask=memory_mask,
            query_position_ids=query_position_ids,
        )

    def forward_projected_bank(
        self,
        hidden_states: torch.Tensor,
        projected_keys: torch.Tensor,
        projected_values: torch.Tensor,
        *,
        memory_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read a bank whose K/V projections were computed when it was written."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must be [B,T,D] with the reader hidden size")
        if hidden_states.shape[1] != 1:
            raise ValueError("cached Bank query must be [B,1,D]")
        if projected_keys.ndim != 4 or projected_keys.shape != projected_values.shape:
            raise ValueError("projected K/V must have matching [B,Hkv,M,Dh] shapes")
        if projected_keys.shape[0] != hidden_states.shape[0]:
            raise ValueError("hidden and projected memory batch sizes differ")
        if projected_keys.shape[1] != self.num_key_value_heads:
            raise ValueError("projected memory has an incompatible KV-head count")
        if projected_keys.shape[-1] != self.head_dim:
            raise ValueError("projected memory has an incompatible head dimension")
        if projected_keys.shape[2] > self.window:
            raise ValueError("cached memory bank exceeds configured window")
        query = self.project_query(
            hidden_states, position_ids=query_position_ids
        )
        output = memory_bank_attention(
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

    def forward_full_bank(
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
        output = strict_past_bank_attention(
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



@dataclass(frozen=True)
class BankBatch:
    """Compact full-sequence bank with original sequence coordinates."""

    memories: torch.Tensor  # [B,M,D], padded chronologically per example
    valid: torch.Tensor  # bool [B,M]
    writes_before: torch.Tensor  # [B,T]
    memory_positions: torch.Tensor  # integer [B,M], original sequence positions
    query_positions: torch.Tensor  # integer [B,T], original sequence positions

    def __post_init__(self) -> None:
        if self.memories.ndim != 3:
            raise ValueError("BankBatch.memories must be [B,M,D]")
        if self.valid.shape != self.memories.shape[:2] or self.valid.dtype != torch.bool:
            raise ValueError("BankBatch.valid must be bool [B,M]")
        if self.writes_before.ndim != 2:
            raise ValueError("BankBatch.writes_before must be [B,T]")
        if self.writes_before.shape[0] != self.memories.shape[0]:
            raise ValueError("bank batch sizes differ")
        if self.memory_positions.shape != self.valid.shape or (
            self.memory_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("BankBatch.memory_positions must be integer [B,M]")
        if self.query_positions.shape != self.writes_before.shape or (
            self.query_positions.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("BankBatch.query_positions must be integer [B,T]")
        if bool((self.memory_positions[self.valid] < 0).any()):
            raise ValueError("valid BankBatch memory positions must be non-negative")
        if bool((self.query_positions < 0).any()):
            raise ValueError("BankBatch query positions must be non-negative")


@dataclass(frozen=True)
class BankCoreRun:
    """Decoder result with an optional internal-layer recurrence source."""

    hidden_states: torch.Tensor
    past_key_values: tuple[LayerKVCache, ...] | None
    captured_hidden: torch.Tensor | None = None


class BankWriter(nn.Module):
    """Minimal learned D->D storage transform, identity-initialized."""

    def __init__(self, hidden_size: int):
        super().__init__()
        # Linear's constructor consumes RNG before overwrite. Forking prevents
        # architecture construction from perturbing experiment/data RNG state.
        with torch.random.fork_rng(devices=[]):
            self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
            with torch.no_grad():
                nn.init.eye_(self.proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("writer input must be [B,T,D]")
        return self.proj(hidden_states)


class BankVariant(MultiPassVariant):
    """Learned bank memory with dense, periodic, or explicit-memory-token writes.

    Dense mode writes every top state; periodic mode writes selected ordinary-token top states. Memory-token mode
    treats ID ``backbone.config.vocab_size`` as an input-only ``<MEM>`` control
    position with its own learned embedding; that ID is never an LM output
    class. A MEM state predicts nothing and writes exactly one bank record.
    """

    variant_name = "bank"
    supports_cached_feedback = True
    supports_bank_nmp = True

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
        super().__init__(backbone)
        if memory_window <= 0:
            raise ValueError("memory_window must be positive")
        if memory_write_mode not in MEMORY_WRITE_MODES:
            raise ValueError("memory_write_mode must be 'dense', 'periodic', or 'memory_token'")
        if memory_write_stride <= 0:
            raise ValueError("memory_write_stride must be positive")
        if memory_token_visibility not in MEMORY_TOKEN_VISIBILITIES:
            raise ValueError("memory_token_visibility must be 'visible' or 'write_only'")
        if memory_write_mode != "memory_token" and memory_token_visibility != "visible":
            raise ValueError("memory_token_visibility applies only to memory_token mode")
        if memory_position_encoding not in MEMORY_POSITION_ENCODINGS:
            raise ValueError("memory_position_encoding must be 'rope' or 'none'")

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
        self.memory_token_visibility = str(memory_token_visibility)
        self.memory_layers = selected_layers
        self.memory_position_encoding = str(memory_position_encoding)
        self.memory_token_id = base_vocab if memory_write_mode == "memory_token" else None
        self.base_vocab_size = base_vocab

        hidden_size = int(backbone.config.hidden_size)
        self.writer = BankWriter(hidden_size)
        if self.memory_write_mode == "memory_token":
            # Zero-init is deliberately conservative: the control slot begins
            # without adding lexical content but can contextualize through the
            # transformer's attention and learns in Phase A as an added param.
            self.memory_token_embedding = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("memory_token_embedding", None)
        self.memory_readers = nn.ModuleDict(
            {
                str(layer_index): BankReader(
                    backbone,
                    window=self.memory_window,
                    position_encoding=self.memory_position_encoding,
                    initialization_seed=int(initialization_seed) + layer_index,
                )
                for layer_index in self.memory_layers
            }
        )
        self._reader_cache_index = {
            layer_index: cache_index
            for cache_index, layer_index in enumerate(self.memory_layers)
        }

    @property
    def uses_memory_tokens(self) -> bool:
        return self.memory_write_mode == "memory_token"

    def phase_a_first_pass_requires_grad(self) -> bool:
        return self.uses_memory_tokens

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().added_parameters()
        yield from self.writer.parameters()
        if self.memory_token_embedding is not None:
            yield self.memory_token_embedding
        yield from self.memory_readers.parameters()

    def memory_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if not self.uses_memory_tokens:
            return torch.zeros_like(input_ids, dtype=torch.bool)
        assert self.memory_token_id is not None
        return input_ids.eq(self.memory_token_id)

    def control_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.memory_token_mask(input_ids)

    def prediction_hidden_after_sequence(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("hidden_states/input_ids token shapes differ")
        if not self.uses_memory_tokens:
            return hidden_states[:, -1:, :]
        ordinary = ~self.memory_token_mask(input_ids)
        if bool((ordinary.sum(dim=1) == 0).any()):
            raise ValueError("memory-token sequence has no linguistic position")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        last = torch.where(ordinary, positions, torch.full_like(positions, -1)).max(dim=1).values
        return hidden_states.gather(
            1, last[:, None, None].expand(-1, 1, hidden_states.shape[-1])
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
            upper = self.base_vocab_size + (1 if self.uses_memory_tokens else 0)
            if bool((input_ids >= upper).any()):
                raise ValueError("input ID lies outside this variant's input vocabulary")
            if not self.uses_memory_tokens and bool((input_ids >= self.base_vocab_size).any()):
                raise ValueError("periodic bank variants accept only ordinary vocabulary IDs")

    def input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        if not self.uses_memory_tokens:
            return self.backbone.model.embed_tokens(input_ids)
        assert self.memory_token_id is not None and self.memory_token_embedding is not None
        is_mem = input_ids.eq(self.memory_token_id)
        safe_ids = input_ids.masked_fill(is_mem, 0)
        ordinary = self.backbone.model.embed_tokens(safe_ids)
        mem = self.memory_token_embedding.to(dtype=ordinary.dtype)[None, None, :]
        return torch.where(is_mem[:, :, None], mem, ordinary)

    def self_attention_key_mask(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if not self.uses_memory_tokens or self.memory_token_visibility == "visible":
            return None
        # Asymmetric write-only semantics: MEM remains a query and can read its
        # causal prefix, but its K/V is unavailable to every later query.
        return ~self.memory_token_mask(input_ids)

    def build_lm_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.uses_memory_tokens:
            return super().build_lm_labels(input_ids)
        self._validate_input_ids(input_ids)
        is_mem = self.memory_token_mask(input_ids)
        ordinary = ~is_mem
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        sentinel = torch.full((bsz, seq_len), seq_len, device=input_ids.device, dtype=torch.long)
        candidates = torch.where(ordinary, positions[None, :].expand(bsz, -1), sentinel)
        # For each physical position, find the nearest ordinary position strictly
        # to its right. This stays device-side; the former Python reverse scan
        # synchronized once per token on CUDA/MPS.
        suffix_min = torch.flip(
            torch.cummin(torch.flip(candidates, dims=(1,)), dim=1).values,
            dims=(1,),
        )
        next_index = torch.cat(
            (suffix_min[:, 1:], torch.full((bsz, 1), seq_len, device=input_ids.device, dtype=torch.long)),
            dim=1,
        )
        safe_index = next_index.clamp(max=max(seq_len - 1, 0))
        next_token = input_ids.gather(1, safe_index)
        valid = ordinary & next_index.lt(seq_len)
        return torch.where(valid, next_token, torch.full_like(input_ids, -100))

    def write_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        if self.memory_write_mode == "dense":
            return torch.ones_like(input_ids, dtype=torch.bool)
        if self.memory_write_mode == "periodic":
            positions = torch.arange(input_ids.shape[1], device=input_ids.device)
            row = (positions + 1).remainder(self.memory_write_stride).eq(0)
            return row[None, :].expand(input_ids.shape[0], -1)
        return self.memory_token_mask(input_ids)

    def sequence_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return cross-attention coordinates anchored to linguistic sequence positions.

        Ordinary dense/periodic inputs use their physical token positions. In
        memory-token mode, an inserted control position inherits the preceding
        linguistic boundary and therefore does not inflate memory age.
        """
        self._validate_input_ids(input_ids)
        if not self.uses_memory_tokens:
            positions = torch.arange(
                input_ids.shape[1], device=input_ids.device, dtype=torch.long
            )
            return positions[None, :].expand(input_ids.shape[0], -1)
        ordinary = ~self.memory_token_mask(input_ids)
        positions = ordinary.long().cumsum(dim=1) - 1
        return positions.clamp_min(0)

    def nmp_written_states(self, final_bank_source: torch.Tensor) -> torch.Tensor:
        return self.writer(final_bank_source)

    def nmp_write_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.write_mask(input_ids)

    def nmp_sequence_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.sequence_positions(input_ids)

    def next_sequence_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_input_ids(input_ids)
        if self.uses_memory_tokens:
            return (~self.memory_token_mask(input_ids)).sum(dim=1, dtype=torch.long)
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

    def build_bank(
        self,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> BankBatch:
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
        return BankBatch(
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

    def _run_bank_core(
        self,
        token_embeddings: torch.Tensor,
        bank: BankBatch | BankState | None,
        *,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        self_attention_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
        post_layer: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
        capture_layer: int | None = None,
    ) -> BankCoreRun:
        if token_embeddings.ndim != 3:
            raise ValueError("token_embeddings must be [B,T,D]")
        if bank is not None and token_embeddings.shape[0] != bank.memories.shape[0]:
            raise ValueError("token and bank batch sizes differ")
        if bank is not None and token_embeddings.shape[-1] != bank.memories.shape[-1]:
            raise ValueError("token and bank hidden dimensions differ")
        cached_bank = isinstance(bank, BankState)
        if cached_bank and token_embeddings.shape[1] != 1:
            raise ValueError("cached bank query must contain exactly one token")
        if cached_bank and bank.capacity != self.memory_window:
            raise ValueError("cached bank capacity differs from memory_window")
        if past_key_values is not None and len(past_key_values) != len(self.backbone.model.layers):
            raise ValueError("past_key_values must contain one cache per layer")

        bsz, seq_len, _ = token_embeddings.shape
        start = 0 if past_key_values is None else self._cache_next_position(past_key_values)
        position_ids = torch.arange(
            start, start + seq_len, device=token_embeddings.device, dtype=torch.long
        )[None, :].expand(bsz, -1)
        if isinstance(bank, BankBatch):
            memory_query_positions = bank.query_positions
        elif isinstance(bank, BankState):
            if query_position_ids is None:
                query_position_ids = bank.next_sequence_positions[:, None]
            if query_position_ids.shape != (bsz, seq_len):
                raise ValueError("cached Bank query positions must be [B,1]")
            memory_query_positions = query_position_ids
        else:
            memory_query_positions = None

        hidden_states = token_embeddings
        new_caches: list[LayerKVCache] | None = [] if use_cache else None
        captured_hidden: torch.Tensor | None = None
        for layer_index, layer in enumerate(self.backbone.model.layers):
            residual = hidden_states
            x = layer.input_layernorm(hidden_states)
            past = None if past_key_values is None else past_key_values[layer_index]
            x, cache = layer.self_attn(
                x,
                attention_mask=self_attention_mask,
                position_ids=position_ids,
                past_key_value=past,
                use_cache=use_cache,
                fast_attention_compatible=(past_key_values is None),
            )
            hidden_states = residual + x
            if new_caches is not None:
                if cache is None:
                    raise RuntimeError("cached bank layer did not return KV state")
                new_caches.append(cache)

            reader_key = str(layer_index)
            if bank is not None and reader_key in self.memory_readers:
                memory_reader = self.memory_readers[reader_key]
                if cached_bank:
                    assert isinstance(bank, BankState)
                    if bank.projected_keys is None:
                        memory_delta = memory_reader.forward_bank(
                            hidden_states,
                            bank.memories,
                            memory_mask=bank.valid,
                            query_position_ids=memory_query_positions,
                            memory_position_ids=bank.positions,
                        )
                    else:
                        if bank.projected_values is None or len(bank.projected_keys) != len(self.memory_readers):
                            raise ValueError("BankState projected K/V does not match bank readers")
                        cache_index = self._reader_cache_index[layer_index]
                        memory_delta = memory_reader.forward_projected_bank(
                            hidden_states,
                            bank.projected_keys[cache_index],
                            bank.projected_values[cache_index],
                            memory_mask=bank.valid,
                            query_position_ids=memory_query_positions,
                        )
                else:
                    assert isinstance(bank, BankBatch)
                    memory_delta = memory_reader.forward_full_bank(
                        hidden_states,
                        bank.memories,
                        writes_before=bank.writes_before,
                        memory_mask=bank.valid,
                        query_position_ids=memory_query_positions,
                        memory_position_ids=bank.memory_positions,
                        dense=(
                            self.memory_write_mode == "dense"
                            or (
                                self.memory_write_mode == "periodic"
                                and self.memory_write_stride == 1
                            )
                        ),
                    )
                hidden_states = hidden_states + memory_delta
            residual = hidden_states
            x = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(x)
            if post_layer is not None:
                hidden_states = post_layer(layer_index, hidden_states)
            if layer_index == capture_layer:
                captured_hidden = hidden_states

        hidden_states = self.backbone.model.norm(hidden_states)
        if capture_layer is not None and captured_hidden is None:
            raise RuntimeError("requested recurrence source layer was not reached")
        return BankCoreRun(
            hidden_states=hidden_states,
            past_key_values=(tuple(new_caches) if new_caches is not None else None),
            captured_hidden=captured_hidden,
        )

    def _run_bank_feedback_core(
        self,
        token_embeddings: torch.Tensor,
        bank: BankBatch | BankState,
        *,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        self_attention_mask: torch.Tensor | None = None,
        query_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...] | None]:
        """Compatibility wrapper for a bank-only decoder pass."""
        run = self._run_bank_core(
            token_embeddings,
            bank,
            past_key_values=past_key_values,
            use_cache=use_cache,
            self_attention_mask=self_attention_mask,
            query_position_ids=query_position_ids,
        )
        return run.hidden_states, run.past_key_values

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        bank = self.build_bank(previous_hidden, input_ids)
        hidden, _ = self._run_bank_feedback_core(
            token_embeddings,
            bank,
            past_key_values=None,
            use_cache=False,
            self_attention_mask=self.self_attention_key_mask(input_ids),
        )
        return hidden

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        bank = self.build_bank(previous_hidden, input_ids)
        hidden, cache = self._run_bank_feedback_core(
            token_embeddings,
            bank,
            past_key_values=None,
            use_cache=True,
            self_attention_mask=self.self_attention_key_mask(input_ids),
        )
        if cache is None:
            raise RuntimeError("cached Bank prefill did not return KV state")
        return hidden, cache

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: BankState,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        if not isinstance(feedback_memory, BankState):
            raise TypeError("Bank cached feedback requires BankState")
        if token is None:
            raise ValueError("cached Bank requires the current token ID")
        query_positions = self._cached_query_positions(feedback_memory, token)
        hidden, cache = self._run_bank_feedback_core(
            token_embedding,
            feedback_memory,
            past_key_values=past_key_values,
            use_cache=True,
            self_attention_mask=self.self_attention_key_mask(token),
            query_position_ids=query_positions,
        )
        if cache is None:
            raise RuntimeError("cached Bank token did not return KV state")
        return hidden, cache

    def _cached_query_positions(
        self, state: BankState, token: torch.Tensor
    ) -> torch.Tensor:
        if token.shape != (state.batch_size, 1):
            raise ValueError("cached Bank token must be [B,1]")
        query_positions = state.next_sequence_positions[:, None]
        if self.uses_memory_tokens:
            query_positions = torch.where(
                self.memory_token_mask(token),
                (query_positions - 1).clamp_min(0),
                query_positions,
            )
        return query_positions

    def _state_from_bank_batch(self, bank: BankBatch) -> BankState:
        bsz, _, dim = bank.memories.shape
        result = bank.memories.new_zeros((bsz, self.memory_window, dim))
        valid = torch.zeros((bsz, self.memory_window), dtype=torch.bool, device=bank.memories.device)
        positions = torch.zeros(
            (bsz, self.memory_window), dtype=torch.long, device=bank.memories.device
        )
        for batch_index in range(bsz):
            row = bank.memories[batch_index, bank.valid[batch_index], :][-self.memory_window :]
            row_positions = bank.memory_positions[
                batch_index, bank.valid[batch_index]
            ][-self.memory_window :]
            count = row.shape[0]
            if count:
                result[batch_index, :count, :] = row
                valid[batch_index, :count] = True
                positions[batch_index, :count] = row_positions
        next_positions = bank.query_positions[:, -1] + 1
        return self._project_state(result, valid, positions, next_positions)

    def _project_state(
        self,
        memories: torch.Tensor,
        valid: torch.Tensor,
        positions: torch.Tensor,
        next_sequence_positions: torch.Tensor,
    ) -> BankState:
        projected_keys: list[torch.Tensor] = []
        projected_values: list[torch.Tensor] = []
        for reader in self.memory_readers.values():
            key, value = reader.project_memory(
                memories, position_ids=positions
            )
            projected_keys.append(key.detach())
            projected_values.append(value.detach())
        return BankState(
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
    ) -> BankState:
        if hidden_states.ndim != 3 or hidden_states.shape[1] < 1:
            raise ValueError("hidden_states must be non-empty [B,T,D]")
        if input_ids is None:
            if self.memory_write_mode == "memory_token":
                raise ValueError("memory-token mode requires input_ids to seed feedback memory")
            input_ids = torch.zeros(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)
        return self._state_from_bank_batch(self.build_bank(hidden_states, input_ids))

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
                raise ValueError("periodic cached write requires absolute position")
            trigger = (int(position) + 1) % self.memory_write_stride == 0
            return torch.full((token.shape[0],), trigger, dtype=torch.bool, device=token.device)
        assert self.memory_token_id is not None
        return token[:, 0].eq(self.memory_token_id)

    def _append_bank(
        self,
        state: BankState,
        new_hidden: torch.Tensor,
        *,
        trigger: torch.Tensor,
        write_positions: torch.Tensor,
        next_sequence_positions: torch.Tensor,
    ) -> BankState:
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
            raise ValueError("bank and new hidden shapes are incompatible")
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
            raise ValueError("BankState projected K/V does not match bank readers")
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
        return BankState(
            memories=memories.detach(),
            valid=valid.detach(),
            positions=positions.detach(),
            next_sequence_positions=next_sequence_positions.detach(),
            projected_keys=tuple(projected_keys),
            projected_values=tuple(projected_values),
        )

    def _append_feedback_memory(
        self,
        feedback_memory: BankState,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> BankState:
        if not isinstance(feedback_memory, BankState):
            raise TypeError("Bank feedback requires BankState")
        trigger = self._write_trigger(token=token, position=position)
        if token is None:
            raise ValueError("Bank feedback update requires current token")
        current_positions = feedback_memory.next_sequence_positions
        next_positions = current_positions + 1
        if self.uses_memory_tokens:
            is_memory = self.memory_token_mask(token)[:, 0]
            write_positions = torch.where(
                is_memory, (current_positions - 1).clamp_min(0), current_positions
            )
            next_positions = torch.where(
                is_memory, current_positions, next_positions
            )
        else:
            write_positions = current_positions
        return self._append_bank(
            feedback_memory,
            new_hidden,
            trigger=trigger,
            write_positions=write_positions,
            next_sequence_positions=next_positions,
        )
