"""Configuration-aware dominant FLOP estimates for training comparisons.

The estimator intentionally counts the large matrix products that dominate
Transformer training.  One multiply-add is counted as two FLOPs.  It includes
the backbone projections, attention score/value products, LM-head projections,
Bank projections/attention, and recurrent controller/projection matrices.

LayerNorm/RMSNorm, activations, softmax, RoPE, masking/gathering, residual
adds, embedding lookups, and optimizer bookkeeping are not assigned synthetic
costs.  Those costs are implementation- and kernel-dependent and are small
relative to the counted matrix products for this model.  Backward training
FLOPs use the conventional 3x forward matmul estimate: one forward matmul and
two equivalent matmuls for its input/weight gradients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from tiny_mistral.config import MistralConfig


FLOPS_PER_MATMUL = 2
BACKWARD_MULTIPLIER = 3


def _linear_flops(tokens: int, input_features: int, output_features: int) -> int:
    """FLOPs for a bias-free dense matrix multiplication."""
    return (
        FLOPS_PER_MATMUL
        * int(tokens)
        * int(input_features)
        * int(output_features)
    )


def _validate_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def memory_token_layout(
    linguistic_length: int,
    interval: int,
) -> tuple[bool, ...]:
    """Return ``True`` at physical positions occupied by a memory token."""
    linguistic_length = _validate_positive("linguistic_length", linguistic_length)
    interval = _validate_positive("interval", interval)
    layout: list[bool] = []
    remaining = linguistic_length
    while remaining:
        count = min(interval, remaining)
        layout.extend([False] * count)
        remaining -= count
        if remaining:
            layout.append(True)
    return tuple(layout)


def _causal_pairs(
    key_valid: tuple[bool, ...],
    sliding_window: int | None,
) -> int:
    """Count causal self-attention query/key pairs under the model's mask."""
    sequence_length = len(key_valid)
    if sequence_length <= 0:
        raise ValueError("key_valid must be non-empty")
    if sliding_window is not None:
        sliding_window = _validate_positive("sliding_window", sliding_window)
    pairs = 0
    for query_position in range(sequence_length):
        start = 0 if sliding_window is None else max(
            0, query_position + 1 - sliding_window
        )
        pairs += sum(key_valid[start : query_position + 1])
    return pairs


def _bank_pairs(
    sequence_length: int,
    write_positions: tuple[int, ...],
    memory_window: int,
) -> int:
    """Count strict-past Bank query/memory pairs for a full sequence."""
    sequence_length = _validate_positive("sequence_length", sequence_length)
    memory_window = _validate_positive("memory_window", memory_window)
    if any(position < 0 or position >= sequence_length for position in write_positions):
        raise ValueError("Bank write positions must lie inside the sequence")
    if tuple(sorted(set(write_positions))) != write_positions:
        raise ValueError("Bank write positions must be sorted and unique")
    pairs = 0
    writes_seen = 0
    write_cursor = 0
    for query_position in range(sequence_length):
        while write_cursor < len(write_positions) and write_positions[write_cursor] < query_position:
            writes_seen += 1
            write_cursor += 1
        pairs += min(writes_seen, memory_window)
    return pairs


def bank_write_positions(
    *,
    linguistic_length: int,
    memory_write_mode: str,
    memory_write_stride: int | None = None,
) -> tuple[bool, tuple[int, ...], tuple[bool, ...]]:
    """Return physical layout, write positions, and self-attention key mask."""
    linguistic_length = _validate_positive("linguistic_length", linguistic_length)
    if memory_write_mode not in {"dense", "periodic", "memory_token"}:
        raise ValueError("memory_write_mode must be dense, periodic, or memory_token")

    if memory_write_mode == "memory_token":
        if memory_write_stride is None:
            raise ValueError("memory-token mode requires memory_write_stride")
        layout = memory_token_layout(linguistic_length, memory_write_stride)
        writes = tuple(index for index, is_memory in enumerate(layout) if is_memory)
        return True, writes, layout

    layout = (False,) * linguistic_length
    if memory_write_mode == "dense":
        writes = tuple(range(linguistic_length))
    else:
        if memory_write_stride is None:
            raise ValueError("periodic mode requires memory_write_stride")
        stride = _validate_positive("memory_write_stride", memory_write_stride)
        writes = tuple(
            position
            for position in range(linguistic_length)
            if (position + 1) % stride == 0
        )
    return False, writes, layout


@dataclass(frozen=True, slots=True)
class FlopBreakdown:
    """Forward dominant-matmul FLOPs for one full model pass."""

    self_attention_projections: int = 0
    self_attention_products: int = 0
    mlp_projections: int = 0
    lm_head: int = 0
    bank_writer: int = 0
    bank_reader_projections: int = 0
    bank_reader_products: int = 0
    recurrent_controller: int = 0
    recurrent_projection: int = 0

    @property
    def total(self) -> int:
        return sum(asdict(self).values())

    def __add__(self, other: "FlopBreakdown") -> "FlopBreakdown":
        if not isinstance(other, FlopBreakdown):
            return NotImplemented
        values = {
            field: getattr(self, field) + getattr(other, field)
            for field in asdict(self)
        }
        return FlopBreakdown(**values)

    def scaled(self, factor: float) -> dict[str, float]:
        return {
            field: float(getattr(self, field)) * float(factor)
            for field in asdict(self)
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total"] = self.total
        return payload


@dataclass(frozen=True, slots=True)
class PassFlopEstimate:
    """Forward and training FLOPs for a fixed pass count."""

    variant: str
    passes: int
    linguistic_sequence_length: int
    physical_sequence_length: int
    memory_positions: int
    bank_write_positions: int
    forward: FlopBreakdown

    @property
    def forward_flops(self) -> int:
        return self.forward.total

    @property
    def training_flops(self) -> int:
        return BACKWARD_MULTIPLIER * self.forward_flops

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "passes": self.passes,
            "linguistic_sequence_length": self.linguistic_sequence_length,
            "physical_sequence_length": self.physical_sequence_length,
            "memory_positions": self.memory_positions,
            "bank_write_positions": self.bank_write_positions,
            "forward": self.forward.to_dict(),
            "forward_flops": self.forward_flops,
            "training_flops": self.training_flops,
        }


@dataclass(frozen=True, slots=True)
class ScheduledFlopEstimate:
    """Weighted FLOPs for a pass-count schedule relative to vanilla K=1."""

    variant: str
    pass_probabilities: dict[int, float]
    pass_estimates: dict[int, PassFlopEstimate]
    weighted_forward_flops: float
    weighted_training_flops: float
    baseline_training_flops: int
    relative_training_flops: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "pass_probabilities": {
                str(key): value for key, value in self.pass_probabilities.items()
            },
            "pass_estimates": {
                str(key): value.to_dict() for key, value in self.pass_estimates.items()
            },
            "weighted_forward_flops": self.weighted_forward_flops,
            "weighted_training_flops": self.weighted_training_flops,
            "baseline_training_flops": self.baseline_training_flops,
            "relative_training_flops": self.relative_training_flops,
        }


def _backbone_pass_breakdown(
    config: MistralConfig,
    *,
    key_valid: tuple[bool, ...],
) -> FlopBreakdown:
    sequence_length = len(key_valid)
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    kv_width = int(config.num_key_value_heads) * int(config.head_dim)
    layers = int(config.num_hidden_layers)

    self_attention_projections = layers * (
        _linear_flops(sequence_length, hidden_size, hidden_size)
        + 2 * _linear_flops(sequence_length, hidden_size, kv_width)
        + _linear_flops(sequence_length, hidden_size, hidden_size)
    )
    self_attention_products = layers * (
        FLOPS_PER_MATMUL
        * int(hidden_size)
        * _causal_pairs(key_valid, config.sliding_window)
        * 2
    )
    mlp_projections = layers * (
        2 * _linear_flops(sequence_length, hidden_size, intermediate_size)
        + _linear_flops(sequence_length, intermediate_size, hidden_size)
    )
    lm_head = _linear_flops(sequence_length, hidden_size, int(config.vocab_size))
    return FlopBreakdown(
        self_attention_projections=self_attention_projections,
        self_attention_products=self_attention_products,
        mlp_projections=mlp_projections,
        lm_head=lm_head,
    )


def _bank_breakdown(
    config: MistralConfig,
    *,
    sequence_length: int,
    write_positions: tuple[int, ...],
    memory_window: int,
    reader_layers: int,
) -> FlopBreakdown:
    hidden_size = int(config.hidden_size)
    kv_width = int(config.num_key_value_heads) * int(config.head_dim)
    memory_length = len(write_positions)
    pairs = _bank_pairs(sequence_length, write_positions, memory_window)
    reader_projections = (
        _linear_flops(sequence_length, hidden_size, hidden_size)
        + 2 * _linear_flops(memory_length, hidden_size, kv_width)
        + _linear_flops(sequence_length, hidden_size, hidden_size)
    )
    reader_products = FLOPS_PER_MATMUL * hidden_size * pairs * 2
    return FlopBreakdown(
        bank_writer=_linear_flops(memory_length, hidden_size, hidden_size),
        bank_reader_projections=int(reader_layers) * reader_projections,
        bank_reader_products=int(reader_layers) * reader_products,
    )


def _recurrent_breakdown(
    config: MistralConfig,
    *,
    sequence_length: int,
    variant: str,
    adaptive_recirculation: bool,
) -> FlopBreakdown:
    hidden_size = int(config.hidden_size)
    if variant == "memory_add":
        return FlopBreakdown(
            recurrent_projection=_linear_flops(sequence_length, hidden_size, hidden_size)
        )
    if variant in {"recirculation", "bank_recirculation_hybrid"}:
        if not adaptive_recirculation:
            return FlopBreakdown()
        return FlopBreakdown(
            recurrent_controller=(
                _linear_flops(sequence_length, 2 * hidden_size, hidden_size)
                + _linear_flops(sequence_length, hidden_size, hidden_size)
                + _linear_flops(sequence_length, hidden_size, 2 * hidden_size)
            )
        )
    if variant == "bank_add_hybrid":
        return FlopBreakdown(
            recurrent_projection=_linear_flops(sequence_length, hidden_size, hidden_size)
        )
    return FlopBreakdown()


def estimate_pass(
    config: MistralConfig,
    *,
    variant: str,
    passes: int,
    linguistic_sequence_length: int,
    memory_window: int = 32,
    memory_write_mode: str | None = None,
    memory_write_stride: int | None = None,
    memory_token_visibility: str = "visible",
    memory_layers: Iterable[int] | str = "all",
    recirculation_mode: str = "fixed",
) -> PassFlopEstimate:
    """Estimate forward/training FLOPs for one fixed K-pass optimizer step."""
    passes = _validate_positive("passes", passes)
    linguistic_sequence_length = _validate_positive(
        "linguistic_sequence_length", linguistic_sequence_length
    )
    if variant not in {
        "vanilla",
        "memory_add",
        "recirculation",
        "bank",
        "bank_add_hybrid",
        "bank_recirculation_hybrid",
    }:
        raise ValueError(f"unsupported variant {variant!r}")
    if variant == "vanilla" and passes != 1:
        raise ValueError("vanilla FLOP estimation only supports passes=1")
    if variant != "vanilla" and passes < 2:
        raise ValueError("research variants require at least two passes")

    uses_bank = variant in {"bank", "bank_add_hybrid", "bank_recirculation_hybrid"}
    if uses_bank and memory_write_mode is None:
        raise ValueError("Bank variants require memory_write_mode")
    if not uses_bank and memory_write_mode is not None:
        raise ValueError("memory_write_mode only applies to Bank variants")

    if uses_bank:
        uses_control_tokens, writes, layout = bank_write_positions(
            linguistic_length=linguistic_sequence_length,
            memory_write_mode=str(memory_write_mode),
            memory_write_stride=memory_write_stride,
        )
        physical_length = len(layout)
        if uses_control_tokens:
            if memory_token_visibility not in {"visible", "write_only"}:
                raise ValueError(
                    "memory_token_visibility must be visible or write_only"
                )
            key_valid = (
                tuple(not is_memory for is_memory in layout)
                if memory_token_visibility == "write_only"
                else (True,) * physical_length
            )
        else:
            key_valid = (True,) * physical_length
    else:
        writes = ()
        physical_length = linguistic_sequence_length
        key_valid = (True,) * physical_length

    base = _backbone_pass_breakdown(config, key_valid=key_valid)
    if uses_bank:
        assert memory_write_mode is not None
        bank = _bank_breakdown(
            config,
            sequence_length=physical_length,
            write_positions=writes,
            memory_window=memory_window,
            reader_layers=(
                int(config.num_hidden_layers)
                if memory_layers == "all"
                else len(tuple(memory_layers))
            ),
        )
    else:
        bank = FlopBreakdown()

    recurrent = _recurrent_breakdown(
        config,
        sequence_length=physical_length,
        variant=variant,
        adaptive_recirculation=recirculation_mode == "adaptive",
    )
    per_pass_extra = bank + recurrent
    total = FlopBreakdown()
    for pass_index in range(passes):
        total = total + base
        if pass_index > 0:
            total = total + per_pass_extra
    return PassFlopEstimate(
        variant=variant,
        passes=passes,
        linguistic_sequence_length=linguistic_sequence_length,
        physical_sequence_length=physical_length,
        memory_positions=physical_length - linguistic_sequence_length,
        bank_write_positions=len(writes),
        forward=total,
    )


def estimate_schedule(
    config: MistralConfig,
    *,
    variant: str,
    pass_probabilities: Mapping[int, float],
    linguistic_sequence_length: int,
    memory_window: int = 32,
    memory_write_mode: str | None = None,
    memory_write_stride: int | None = None,
    memory_token_visibility: str = "visible",
    memory_layers: Iterable[int] | str = "all",
    recirculation_mode: str = "fixed",
) -> ScheduledFlopEstimate:
    """Estimate a pass schedule and normalize it to vanilla K=1."""
    if not pass_probabilities:
        raise ValueError("pass_probabilities must be non-empty")
    probabilities = {int(key): float(value) for key, value in pass_probabilities.items()}
    if any(key <= 0 or value < 0 for key, value in probabilities.items()):
        raise ValueError("pass counts must be positive and probabilities non-negative")
    total_probability = sum(probabilities.values())
    if total_probability <= 0:
        raise ValueError("pass_probabilities must contain positive mass")
    probabilities = {
        key: value / total_probability for key, value in sorted(probabilities.items())
    }

    estimates = {
        passes: estimate_pass(
            config,
            variant=variant,
            passes=passes,
            linguistic_sequence_length=linguistic_sequence_length,
            memory_window=memory_window,
            memory_write_mode=memory_write_mode,
            memory_write_stride=memory_write_stride,
            memory_token_visibility=memory_token_visibility,
            memory_layers=memory_layers,
            recirculation_mode=recirculation_mode,
        )
        for passes in probabilities
    }
    weighted_forward = sum(
        probabilities[passes] * estimate.forward_flops
        for passes, estimate in estimates.items()
    )
    weighted_training = BACKWARD_MULTIPLIER * weighted_forward
    baseline = estimate_pass(
        config,
        variant="vanilla",
        passes=1,
        linguistic_sequence_length=linguistic_sequence_length,
    ).training_flops
    return ScheduledFlopEstimate(
        variant=variant,
        pass_probabilities=probabilities,
        pass_estimates=estimates,
        weighted_forward_flops=weighted_forward,
        weighted_training_flops=weighted_training,
        baseline_training_flops=baseline,
        relative_training_flops=weighted_training / baseline,
    )
