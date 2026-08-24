"""Experiment configuration and public Memory Attention variant aliases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import yaml

from .nmp import NMP_TARGET_NORMALIZATIONS

# The implementation still uses historical ``bank`` names internally so
# existing checkpoints, configs, and result paths remain loadable. New
# experiments can use the clearer Memory Attention names below.

MEMORY_ATTENTION_VARIANT_ALIASES = {
    "memory_attention": "bank",
    "memory_attention_multiscale": "bank_multiscale",
    "memory_attention_add_hybrid": "bank_add_hybrid",
    "memory_attention_recirculation_hybrid": "bank_recirculation_hybrid",
}


def canonical_variant_name(name: str) -> str:
    """Return the historical implementation name for a public variant alias."""
    return MEMORY_ATTENTION_VARIANT_ALIASES.get(name, name)


SUPPORTED_VARIANTS = {
    "vanilla",
    "fbt",
    "memory_add",
    "recirculation",
    "bank",
    "bank_multiscale",
    "bank_add_hybrid",
    "bank_recirculation_hybrid",
    *MEMORY_ATTENTION_VARIANT_ALIASES,
    "sparse_swa",
}

MEMORY_ATTENTION_VARIANTS = {
    "bank",
    "memory_attention",
    "bank_add_hybrid",
    "memory_attention_add_hybrid",
    "bank_recirculation_hybrid",
    "memory_attention_recirculation_hybrid",
    "bank_multiscale",
    "memory_attention_multiscale",
}
MEMORY_ATTENTION_WRITE_VARIANTS = MEMORY_ATTENTION_VARIANTS - {
    "bank_multiscale",
    "memory_attention_multiscale",
}
MULTISCALE_MEMORY_ATTENTION_VARIANTS = {
    "bank_multiscale",
    "memory_attention_multiscale",
}
SUPPORTED_LR_SCHEDULES = {"constant", "cosine", "piecewise_linear"}
SUPPORTED_AUTOCAST_DTYPES = {"bfloat16"}


def _coerce_pass_probabilities(raw: Any) -> dict[int, float]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("pass-schedule probabilities must be a non-empty mapping")
    result: dict[int, float] = {}
    for key, value in raw.items():
        try:
            passes = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid pass count {key!r}") from exc
        probability = float(value)
        if passes < 1:
            raise ValueError("pass counts must be positive")
        if not math.isfinite(probability) or probability < 0:
            raise ValueError("pass probabilities must be finite and non-negative")
        result[passes] = probability
    total = sum(result.values())
    if total <= 0:
        raise ValueError("pass probabilities must contain positive mass")
    return {passes: probability / total for passes, probability in sorted(result.items())}


def _coerce_pass_loss_weights_by_k(
    raw: Any, *, field_name: str = "pass_loss_weights_by_k"
) -> dict[int, list[float]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    result: dict[int, list[float]] = {}
    for key, values in raw.items():
        try:
            passes = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid pass-loss weight K {key!r}") from exc
        if passes < 1:
            raise ValueError("pass-loss weight K values must be positive")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{field_name}[{passes}] must be a non-empty list")
        weights = [float(value) for value in values]
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("pass-loss weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError(f"{field_name}[{passes}] must contain positive mass")
        result[passes] = weights
    return dict(sorted(result.items()))


def _coerce_early_stop(raw: Any) -> dict[str, Any] | None:
    """Canonicalize validation gates that stop a run when all gates pass."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("early_stop must be a mapping")
    unknown = sorted(
        set(raw)
        - {
            "pass_nll_max",
            "pass_nll_delta_max",
            "hidden_delta_nonincreasing",
        }
    )
    if unknown:
        raise ValueError(f"unknown early_stop fields: {unknown}")

    pass_nll_max_raw = raw.get("pass_nll_max", {})
    if not isinstance(pass_nll_max_raw, dict):
        raise ValueError("early_stop.pass_nll_max must be a mapping")
    pass_nll_max: dict[int, float] = {}
    for key, value in pass_nll_max_raw.items():
        try:
            pass_index = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid early-stop pass index {key!r}") from exc
        threshold = float(value)
        if pass_index < 1:
            raise ValueError("early-stop pass indices must be positive")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("early-stop pass NLL limits must be finite and non-negative")
        pass_nll_max[pass_index] = threshold

    pass_nll_delta_raw = raw.get("pass_nll_delta_max", [])
    if not isinstance(pass_nll_delta_raw, list):
        raise ValueError("early_stop.pass_nll_delta_max must be a list")
    pass_nll_delta_max: list[dict[str, float | int]] = []
    for gate in pass_nll_delta_raw:
        if not isinstance(gate, dict):
            raise ValueError("each pass_nll_delta_max gate must be a mapping")
        gate_unknown = sorted(set(gate) - {"pass", "reference_pass", "max_delta"})
        if gate_unknown:
            raise ValueError(f"unknown pass_nll_delta_max fields: {gate_unknown}")
        if set(gate) != {"pass", "reference_pass", "max_delta"}:
            raise ValueError(
                "pass_nll_delta_max gates require pass, reference_pass, and max_delta"
            )
        pass_index = int(gate["pass"])
        reference_pass = int(gate["reference_pass"])
        max_delta = float(gate["max_delta"])
        if pass_index < 1 or reference_pass < 1:
            raise ValueError("early-stop pass indices must be positive")
        if not math.isfinite(max_delta):
            raise ValueError("early-stop pass NLL deltas must be finite")
        pass_nll_delta_max.append(
            {
                "pass": pass_index,
                "reference_pass": reference_pass,
                "max_delta": max_delta,
            }
        )

    hidden_delta_nonincreasing = raw.get("hidden_delta_nonincreasing", False)
    if not isinstance(hidden_delta_nonincreasing, bool):
        raise ValueError("early_stop.hidden_delta_nonincreasing must be boolean")
    if not pass_nll_max and not pass_nll_delta_max and not hidden_delta_nonincreasing:
        raise ValueError("early_stop must enable at least one validation gate")
    return {
        "pass_nll_max": dict(sorted(pass_nll_max.items())),
        "pass_nll_delta_max": pass_nll_delta_max,
        "hidden_delta_nonincreasing": hidden_delta_nonincreasing,
    }


def _coerce_layer_indices(raw: Any, *, field_name: str) -> str | list[int]:
    """Canonicalize selected layer indices while retaining an ``all`` shorthand."""
    if raw == "all":
        return "all"
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{field_name} must be 'all' or a non-empty list of indices")
    layers = [int(value) for value in raw]
    if any(layer < 0 for layer in layers):
        raise ValueError(f"{field_name} indices must be non-negative")
    if len(layers) != len(set(layers)):
        raise ValueError(f"{field_name} indices must be unique")
    return sorted(layers)


def normalize_pass_schedule(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Validate and normalize a token-indexed pass-count schedule.

    Each stage has ``probabilities`` and may have an exclusive ``until_tokens``
    bound. The last stage must be unbounded. Example::

        [{"until_tokens": 1_000_000, "probabilities": {2: 1.0}},
         {"probabilities": {1: .5, 2: .45, 3: .05}}]
    """
    if raw is None:
        return [{"until_tokens": None, "probabilities": {1: 1.0}}]
    if not isinstance(raw, list) or not raw:
        raise ValueError("pass_schedule must be a non-empty list")
    stages: list[dict[str, Any]] = []
    previous_until = 0
    for index, stage in enumerate(raw):
        if not isinstance(stage, dict):
            raise ValueError("each pass_schedule stage must be a mapping")
        unknown = sorted(set(stage) - {"until_tokens", "probabilities"})
        if unknown:
            raise ValueError(f"unknown pass_schedule stage fields: {unknown}")
        until = stage.get("until_tokens")
        if until is not None:
            until = int(until)
            if until <= previous_until:
                raise ValueError("pass_schedule until_tokens must increase strictly")
            previous_until = until
        elif index != len(raw) - 1:
            raise ValueError("only the final pass_schedule stage may omit until_tokens")
        stages.append(
            {
                "until_tokens": until,
                "probabilities": _coerce_pass_probabilities(stage.get("probabilities")),
            }
        )
    if stages[-1]["until_tokens"] is not None:
        raise ValueError("the final pass_schedule stage must be unbounded")
    return stages


def validate_lr_schedule(raw: dict[str, Any] | None) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ValueError("lr_schedule must be a mapping")
    schedule_type = str(raw.get("type", "cosine"))
    if schedule_type not in SUPPORTED_LR_SCHEDULES:
        raise ValueError(f"unsupported lr_schedule type {schedule_type!r}")
    if schedule_type == "constant":
        unknown = sorted(set(raw) - {"type"})
        if unknown:
            raise ValueError(f"unknown constant lr_schedule fields: {unknown}")
        return
    if schedule_type == "cosine":
        unknown = sorted(set(raw) - {"type", "warmup_tokens", "min_multiplier"})
        if unknown:
            raise ValueError(f"unknown cosine lr_schedule fields: {unknown}")
        warmup = int(raw.get("warmup_tokens", 0))
        minimum = float(raw.get("min_multiplier", 0.1))
        if warmup < 0 or not 0 <= minimum <= 1:
            raise ValueError("cosine schedule requires warmup_tokens>=0 and min_multiplier in [0,1]")
        return
    unknown = sorted(set(raw) - {"type", "points"})
    if unknown:
        raise ValueError(f"unknown piecewise_linear lr_schedule fields: {unknown}")
    points = raw.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("piecewise_linear schedule requires non-empty points")
    last_token = -1
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("piecewise_linear points must be [tokens, multiplier] pairs")
        tokens, multiplier = int(point[0]), float(point[1])
        if tokens < 0 or tokens <= last_token:
            raise ValueError("piecewise_linear token coordinates must increase from >=0")
        if not math.isfinite(multiplier) or multiplier < 0:
            raise ValueError("piecewise_linear multipliers must be finite and non-negative")
        last_token = tokens


@dataclass(slots=True)
class ExperimentConfig:
    variant: str = "vanilla"
    model_dir: str = "checkpoints/TinyMistral-248M-v3"
    data_dir: str = "data/dolmino/wiring_2048"
    output_dir: str = "benchmarks/controls/smoke/results/vanilla"
    device: str = "auto"
    dtype: str = "float32"
    autocast_dtype: str | None = None
    attention_backend: str = "auto"
    seed: int = 1337
    architecture_seed: int = 4242
    batch_size: int = 1
    grad_accum_steps: int = 1
    max_unique_tokens: int = 65_536

    # Base learning rate; Phase-B parameter groups may override it independently.
    learning_rate: float = 1e-6
    pretrained_learning_rate: float | None = None
    added_learning_rate: float | None = None
    min_lr_ratio: float = 0.1
    warmup_tokens: int = 0
    lr_schedule: dict[str, Any] | None = None

    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every_tokens: int = 32_768
    eval_batches: int = 16
    eval_passes: int = 1
    early_stop: dict[str, Any] | None = None
    checkpoint_every_tokens: int = 65_536
    checkpoint_every_seconds: float = 0.0
    checkpoint_keep_last: int = 2
    snapshot_at_tokens: list[int] | None = None

    # Architecture/training protocol knobs. NTP names are explicit because
    # recurrent and Memory Attention NMP have independent pass-loss weightings
    # below. Historical bank field names remain serialized for compatibility.
    phase: str = "B"
    pass_schedule: list[dict[str, Any]] | None = None
    ntp_pass_loss_weights: list[float] | None = None
    ntp_pass_loss_weights_by_k: dict[int, list[float]] | None = None
    # Deprecated aliases accepted for old configs and checkpoints. They are
    # canonicalized to the explicit NTP fields and omitted from to_dict().
    pass_loss_weights: list[float] | None = None
    pass_loss_weights_by_k: dict[int, list[float]] | None = None
    memory_window: int = 32
    # Memory Attention architecture axes. Experiment configs declare them
    # explicitly; historical bank field names remain serialized, and the model
    # constructors retain small ergonomic defaults for unit tests.
    memory_write_mode: str | None = None
    memory_write_stride: int | None = None
    memory_token_visibility: str | None = None
    memory_layers: str | list[int] | None = None
    memory_position_encoding: str | None = None
    # Multiscale Memory Attention retains a dense recent region plus fixed-periodic old
    # records from the same dense previous-pass source stream.
    memory_dense_window: int | None = None
    memory_sparse_window: int | None = None
    memory_sparse_stride: int | None = None
    # Parameter-free one-pass control that widens selected self-attention masks.
    sparse_attention_stride: int | None = None
    sparse_attention_window: int | None = None
    sparse_attention_layers: str | list[int] | None = None
    prefix_mixin_probability: float = 0.0
    fbt_normalize_gate_input: bool = False
    fbt_latent_jitter_std: float = 0.0
    recirculation_source_layer: int | None = None
    recirculation_destination_layer: int | None = None
    recirculation_alpha: float = 0.1
    recirculation_mode: str = "fixed"

    # Training-only next-memory prediction (NMP). Disabled objectives do not
    # instantiate heads and therefore preserve historical model state exactly.
    recurrent_nmp_weight: float = 0.0
    bank_nmp_weight: float = 0.0
    # Recirculation sources are high-amplitude residual-stream states, so the
    # default target is parameter-free RMS-normalized. ``none`` remains useful
    # as a raw-state ablation. Memory Attention targets are always the
    # post-writer memory
    # representation and are intentionally not normalized here.
    recurrent_nmp_target_normalization: str = "rms"
    recurrent_nmp_pass_loss_weights_by_k: dict[int, list[float]] | None = None
    bank_nmp_pass_loss_weights_by_k: dict[int, list[float]] | None = None
    nmp_projection_factor: float = 1.3
    nmp_warmup_tokens: int = 0

    # ``resume_from`` restores the exact run. ``init_from`` loads model weights
    # only and begins a fresh trajectory/optimizer/data schedule.
    resume_from: str | None = None
    init_from: str | None = None

    def __post_init__(self) -> None:
        if self.ntp_pass_loss_weights is not None and self.pass_loss_weights is not None:
            raise ValueError(
                "ntp_pass_loss_weights and legacy pass_loss_weights are mutually exclusive"
            )
        if (
            self.ntp_pass_loss_weights_by_k is not None
            and self.pass_loss_weights_by_k is not None
        ):
            raise ValueError(
                "ntp_pass_loss_weights_by_k and legacy pass_loss_weights_by_k are mutually exclusive"
            )
        if self.ntp_pass_loss_weights is None:
            self.ntp_pass_loss_weights = self.pass_loss_weights
        if self.ntp_pass_loss_weights_by_k is None:
            self.ntp_pass_loss_weights_by_k = self.pass_loss_weights_by_k
        for field_name in (
            "ntp_pass_loss_weights_by_k",
            "recurrent_nmp_pass_loss_weights_by_k",
            "bank_nmp_pass_loss_weights_by_k",
        ):
            value = getattr(self, field_name)
            if value is not None:
                setattr(
                    self,
                    field_name,
                    _coerce_pass_loss_weights_by_k(value, field_name=field_name),
                )
        if self.pass_loss_weights_by_k is not None:
            self.pass_loss_weights_by_k = self.ntp_pass_loss_weights_by_k
        if self.snapshot_at_tokens is not None:
            self.snapshot_at_tokens = sorted({int(value) for value in self.snapshot_at_tokens})
        self.early_stop = _coerce_early_stop(self.early_stop)
        if self.variant in MEMORY_ATTENTION_VARIANTS:
            self.memory_layers = _coerce_layer_indices(
                "all" if self.memory_layers is None else self.memory_layers,
                field_name="memory_layers",
            )
            if self.memory_position_encoding is None:
                self.memory_position_encoding = "rope"
            if (
                self.variant in MULTISCALE_MEMORY_ATTENTION_VARIANTS
                and self.memory_dense_window is not None
                and self.memory_sparse_window is not None
                and self.memory_dense_window >= 0
                and self.memory_sparse_window >= 0
                and self.memory_dense_window + self.memory_sparse_window > 0
            ):
                self.memory_window = (
                    self.memory_dense_window + self.memory_sparse_window
                )
        if self.variant == "sparse_swa":
            self.sparse_attention_layers = _coerce_layer_indices(
                "all" if self.sparse_attention_layers is None else self.sparse_attention_layers,
                field_name="sparse_attention_layers",
            )

    def normalized_pass_schedule(self) -> list[dict[str, Any]]:
        return normalize_pass_schedule(self.pass_schedule)

    @property
    def pretrained_lr(self) -> float:
        return self.learning_rate if self.pretrained_learning_rate is None else float(self.pretrained_learning_rate)

    @property
    def added_lr(self) -> float:
        return self.learning_rate if self.added_learning_rate is None else float(self.added_learning_rate)

    def ntp_loss_weights_for_passes(self, passes: int) -> list[float] | None:
        if passes < 1:
            raise ValueError("passes must be positive")
        if self.ntp_pass_loss_weights_by_k is not None:
            try:
                return self.ntp_pass_loss_weights_by_k[passes]
            except KeyError as exc:
                raise ValueError(
                    f"no NTP pass-loss weights configured for sampled K={passes}"
                ) from exc
        return self.ntp_pass_loss_weights

    # Compatibility method retained for callers using the old generic name.
    def loss_weights_for_passes(self, passes: int) -> list[float] | None:
        return self.ntp_loss_weights_for_passes(passes)

    @staticmethod
    def _nmp_loss_weights_for_passes(
        mapping: dict[int, list[float]] | None,
        passes: int,
        *,
        label: str,
    ) -> list[float] | None:
        if passes < 1:
            raise ValueError("passes must be positive")
        if mapping is None:
            # MultiPassVariant normalizes None to uniform weights. Keeping the
            # default implicit avoids duplicating every K in every config.
            return None
        try:
            return mapping[passes]
        except KeyError as exc:
            raise ValueError(f"no {label} pass-loss weights configured for sampled K={passes}") from exc

    def recurrent_nmp_loss_weights_for_passes(self, passes: int) -> list[float] | None:
        return self._nmp_loss_weights_for_passes(
            self.recurrent_nmp_pass_loss_weights_by_k,
            passes,
            label="recurrent NMP",
        )

    def bank_nmp_loss_weights_for_passes(self, passes: int) -> list[float] | None:
        return self._nmp_loss_weights_for_passes(
            self.bank_nmp_pass_loss_weights_by_k,
            passes,
            label="Bank NMP",
        )

    def nmp_weight_scale_at(self, unique_tokens_seen: int) -> float:
        if unique_tokens_seen < 0:
            raise ValueError("unique_tokens_seen must be non-negative")
        if self.nmp_warmup_tokens == 0:
            return 1.0
        return min(float(unique_tokens_seen) / float(self.nmp_warmup_tokens), 1.0)

    def validate(self) -> None:
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(SUPPORTED_VARIANTS)}; got {self.variant!r}")
        if self.phase not in {"A", "B"}:
            raise ValueError("phase must be 'A' or 'B'")
        if self.resume_from and self.init_from:
            raise ValueError("resume_from and init_from are mutually exclusive")
        if self.autocast_dtype is not None:
            if self.autocast_dtype not in SUPPORTED_AUTOCAST_DTYPES:
                raise ValueError(
                    f"autocast_dtype must be one of {sorted(SUPPORTED_AUTOCAST_DTYPES)}"
                )
            if self.dtype != "float32":
                raise ValueError(
                    "autocast training requires dtype=float32 so learned parameters "
                    "and AdamW state remain FP32"
                )
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must be positive")
        if self.max_unique_tokens <= 0:
            raise ValueError("max_unique_tokens must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        for name, value in (
            ("pretrained_learning_rate", self.pretrained_learning_rate),
            ("added_learning_rate", self.added_learning_rate),
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.warmup_tokens < 0:
            raise ValueError("warmup_tokens must be non-negative")
        validate_lr_schedule(self.lr_schedule)
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight_decay must be non-negative and grad_clip positive")
        if self.eval_every_tokens < 0 or self.eval_batches < 0:
            raise ValueError("evaluation cadence/count must be non-negative")
        if self.eval_passes < 1:
            raise ValueError("eval_passes must be positive")
        self.early_stop = _coerce_early_stop(self.early_stop)
        if self.early_stop is not None:
            if self.eval_every_tokens <= 0:
                raise ValueError("early_stop requires a positive eval_every_tokens cadence")
            if self.eval_passes < 2:
                raise ValueError("early_stop requires multipass validation with eval_passes>=2")
            referenced_passes = set(self.early_stop["pass_nll_max"])
            for gate in self.early_stop["pass_nll_delta_max"]:
                referenced_passes.add(int(gate["pass"]))
                referenced_passes.add(int(gate["reference_pass"]))
            if referenced_passes and max(referenced_passes) > self.eval_passes:
                raise ValueError("early_stop references a pass beyond eval_passes")
            if (
                self.early_stop["hidden_delta_nonincreasing"]
                and self.eval_passes < 3
            ):
                raise ValueError(
                    "hidden_delta_nonincreasing requires eval_passes>=3"
                )
        if self.checkpoint_every_tokens < 0:
            raise ValueError("checkpoint_every_tokens must be non-negative")
        if not math.isfinite(float(self.checkpoint_every_seconds)) or self.checkpoint_every_seconds < 0:
            raise ValueError("checkpoint_every_seconds must be finite and non-negative")
        if self.checkpoint_keep_last < 1:
            raise ValueError("checkpoint_keep_last must be at least 1")
        if self.snapshot_at_tokens is not None:
            if any(value <= 0 or value > self.max_unique_tokens for value in self.snapshot_at_tokens):
                raise ValueError("snapshot_at_tokens values must lie in (0, max_unique_tokens]")
        if self.memory_window <= 0:
            raise ValueError("memory_window must be positive")

        for name, value in (
            ("recurrent_nmp_weight", self.recurrent_nmp_weight),
            ("bank_nmp_weight", self.bank_nmp_weight),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.recurrent_nmp_target_normalization not in NMP_TARGET_NORMALIZATIONS:
            raise ValueError(
                "recurrent_nmp_target_normalization must be one of "
                f"{sorted(NMP_TARGET_NORMALIZATIONS)}"
            )
        if (
            not math.isfinite(float(self.nmp_projection_factor))
            or self.nmp_projection_factor <= 0
        ):
            raise ValueError("nmp_projection_factor must be finite and positive")
        if (
            self.nmp_warmup_tokens < 0
            or self.nmp_warmup_tokens > self.max_unique_tokens
        ):
            raise ValueError("nmp_warmup_tokens must lie in [0, max_unique_tokens]")
        nmp_enabled = self.recurrent_nmp_weight > 0 or self.bank_nmp_weight > 0
        if not nmp_enabled and self.nmp_warmup_tokens != 0:
            raise ValueError("nmp_warmup_tokens requires an enabled NMP objective")
        if nmp_enabled and not (self.init_from or self.resume_from):
            raise ValueError(
                "NMP continuation must use init_from or resume_from from an "
                "NTP-trained run"
            )
        if nmp_enabled and self.nmp_warmup_tokens == 0:
            raise ValueError("enabled NMP objectives require a positive nmp_warmup_tokens ramp")
        recurrent_nmp_variants = {
            "memory_add",
            "recirculation",
            "bank_add_hybrid",
            "bank_recirculation_hybrid",
            "memory_attention_add_hybrid",
            "memory_attention_recirculation_hybrid",
        }
        bank_nmp_variants = MEMORY_ATTENTION_VARIANTS
        if self.recurrent_nmp_weight > 0 and self.variant not in recurrent_nmp_variants:
            raise ValueError(f"variant={self.variant} does not support recurrent NMP")
        if self.bank_nmp_weight > 0 and self.variant not in bank_nmp_variants:
            raise ValueError(f"variant={self.variant} does not support bank NMP")

        recirculation_variants = {
            "recirculation",
            "bank_recirculation_hybrid",
            "memory_attention_recirculation_hybrid",
        }
        if self.variant in recirculation_variants:
            if self.recirculation_mode not in {"fixed", "adaptive"}:
                raise ValueError("recirculation_mode must be 'fixed' or 'adaptive'")
            if (
                self.variant == "recirculation"
                and self.phase == "A"
                and self.recirculation_mode == "fixed"
            ):
                raise ValueError(
                    "basic fixed recirculation has no Phase-A parameters; use phase B"
                )
            if (
                self.recirculation_source_layer is None
                or self.recirculation_destination_layer is None
            ):
                raise ValueError(
                    "recirculation requires source and destination layer fields"
                )
            if not (
                0
                <= self.recirculation_destination_layer
                < self.recirculation_source_layer
            ):
                raise ValueError(
                    "recirculation requires destination_layer < source_layer"
                )
            if not math.isfinite(float(self.recirculation_alpha)) or not 0.0 <= float(
                self.recirculation_alpha
            ) <= 1.0:
                raise ValueError("recirculation_alpha must be finite in [0, 1]")
        elif (
            self.recirculation_source_layer is not None
            or self.recirculation_destination_layer is not None
            or self.recirculation_alpha != 0.1
            or self.recirculation_mode != "fixed"
        ):
            raise ValueError(
                "recirculation_* fields apply only to recirculation variants"
            )

        bank_variants = MEMORY_ATTENTION_WRITE_VARIANTS
        if self.variant in bank_variants:
            if self.memory_write_mode not in {"dense", "periodic", "memory_token"}:
                raise ValueError(
                    "bank configs require memory_write_mode: dense|periodic|memory_token"
                )
            if self.memory_write_mode == "dense":
                if self.memory_write_stride is not None:
                    raise ValueError("dense bank must not set memory_write_stride")
                if self.memory_token_visibility is not None:
                    raise ValueError("dense bank must not set memory_token_visibility")
            elif self.memory_write_mode == "periodic":
                if self.memory_write_stride is None or self.memory_write_stride <= 0:
                    raise ValueError("periodic bank requires positive memory_write_stride")
                if self.memory_token_visibility is not None:
                    raise ValueError("memory_token_visibility applies only to memory_token mode")
            else:
                if self.memory_write_stride is None or self.memory_write_stride <= 0:
                    raise ValueError("memory_token bank requires positive memory_write_stride")
                if self.memory_token_visibility not in {"visible", "write_only"}:
                    raise ValueError(
                        "memory_token bank requires memory_token_visibility: visible|write_only"
                    )
            if self.memory_layers is None:
                raise ValueError("bank configs require memory_layers")
            self.memory_layers = _coerce_layer_indices(
                self.memory_layers, field_name="memory_layers"
            )
            if self.memory_position_encoding not in {"rope", "none"}:
                raise ValueError(
                    "bank configs require memory_position_encoding: rope|none"
                )
            if any(
                value is not None
                for value in (
                    self.memory_dense_window,
                    self.memory_sparse_window,
                    self.memory_sparse_stride,
                )
            ):
                raise ValueError("multiscale memory fields require variant=bank_multiscale")
        elif self.variant in MULTISCALE_MEMORY_ATTENTION_VARIANTS:
            if (
                self.memory_write_mode is not None
                or self.memory_write_stride is not None
                or self.memory_token_visibility is not None
            ):
                raise ValueError("bank_multiscale uses dense retention, not memory_write_* fields")
            if self.memory_dense_window is None or self.memory_sparse_window is None:
                raise ValueError(
                    "bank_multiscale requires memory_dense_window and memory_sparse_window"
                )
            if self.memory_dense_window < 0 or self.memory_sparse_window < 0:
                raise ValueError("multiscale memory windows must be non-negative")
            if self.memory_dense_window + self.memory_sparse_window <= 0:
                raise ValueError("bank_multiscale requires at least one non-zero memory window")
            if self.memory_sparse_stride is None or self.memory_sparse_stride <= 0:
                raise ValueError("bank_multiscale requires positive memory_sparse_stride")
            if self.memory_layers is None:
                raise ValueError("bank_multiscale configs require memory_layers")
            self.memory_layers = _coerce_layer_indices(
                self.memory_layers, field_name="memory_layers"
            )
            if self.memory_position_encoding not in {"rope", "none"}:
                raise ValueError(
                    "bank_multiscale requires memory_position_encoding: rope|none"
                )
        elif (
            self.memory_write_mode is not None
            or self.memory_write_stride is not None
            or self.memory_token_visibility is not None
            or self.memory_layers is not None
            or self.memory_position_encoding is not None
            or self.memory_dense_window is not None
            or self.memory_sparse_window is not None
            or self.memory_sparse_stride is not None
        ):
            raise ValueError("memory_* fields are supported only for bank variants")

        if self.variant == "sparse_swa":
            if self.sparse_attention_stride is None or self.sparse_attention_stride <= 0:
                raise ValueError("sparse_swa requires positive sparse_attention_stride")
            if self.sparse_attention_window is None or self.sparse_attention_window <= 0:
                raise ValueError("sparse_swa requires positive sparse_attention_window")
            if self.sparse_attention_layers is None:
                raise ValueError("sparse_swa requires sparse_attention_layers")
            self.sparse_attention_layers = _coerce_layer_indices(
                self.sparse_attention_layers,
                field_name="sparse_attention_layers",
            )
        elif (
            self.sparse_attention_stride is not None
            or self.sparse_attention_window is not None
            or self.sparse_attention_layers is not None
        ):
            raise ValueError("sparse_attention_* fields require variant=sparse_swa")
        if (
            not math.isfinite(float(self.prefix_mixin_probability))
            or not 0.0 <= float(self.prefix_mixin_probability) <= 1.0
        ):
            raise ValueError("prefix_mixin_probability must be finite and in [0, 1]")
        if self.variant != "fbt" and self.prefix_mixin_probability != 0.0:
            raise ValueError(
                "prefix_mixin_probability is currently supported only for variant=fbt"
            )
        if not isinstance(self.fbt_normalize_gate_input, bool):
            raise ValueError("fbt_normalize_gate_input must be boolean")
        if (
            not math.isfinite(float(self.fbt_latent_jitter_std))
            or self.fbt_latent_jitter_std < 0.0
        ):
            raise ValueError("fbt_latent_jitter_std must be finite and non-negative")
        if self.variant != "fbt" and (
            self.fbt_normalize_gate_input or self.fbt_latent_jitter_std != 0.0
        ):
            raise ValueError("fbt_* fields are supported only for variant=fbt")
        schedule = self.normalized_pass_schedule()
        pass_counts = {passes for stage in schedule for passes in stage["probabilities"]}
        single_pass_variants = {"vanilla", "sparse_swa"}
        if self.variant in single_pass_variants and pass_counts != {1}:
            raise ValueError(f"{self.variant} supports only one-pass training")
        if self.variant in single_pass_variants and self.eval_passes != 1:
            raise ValueError(f"{self.variant} supports eval_passes=1 only")
        if self.phase == "A" and self.variant in single_pass_variants:
            raise ValueError(f"{self.variant} has no Phase-A parameters")
        if self.phase == "A" and any(passes < 2 for passes in pass_counts):
            raise ValueError("Phase A for multipass variants requires at least two passes on every batch")
        if self.ntp_pass_loss_weights is not None and self.ntp_pass_loss_weights_by_k is not None:
            raise ValueError(
                "ntp_pass_loss_weights and ntp_pass_loss_weights_by_k are mutually exclusive"
            )
        if self.ntp_pass_loss_weights is not None:
            if not self.ntp_pass_loss_weights:
                raise ValueError("ntp_pass_loss_weights must not be empty")
            weights = [float(value) for value in self.ntp_pass_loss_weights]
            if any(not math.isfinite(value) or value < 0 for value in weights):
                raise ValueError("ntp_pass_loss_weights must be finite and non-negative")
            if sum(weights) <= 0:
                raise ValueError("ntp_pass_loss_weights must contain positive mass")
        if self.ntp_pass_loss_weights_by_k is not None:
            configured = set(self.ntp_pass_loss_weights_by_k)
            if configured != pass_counts:
                raise ValueError(
                    "ntp_pass_loss_weights_by_k keys must exactly match sampled pass counts "
                    f"{sorted(pass_counts)}; got {sorted(configured)}"
                )
            for passes, weights in self.ntp_pass_loss_weights_by_k.items():
                if len(weights) != passes:
                    raise ValueError(
                        f"ntp_pass_loss_weights_by_k[{passes}] must contain exactly {passes} weights"
                    )
        for name, mapping in (
            ("recurrent_nmp_pass_loss_weights_by_k", self.recurrent_nmp_pass_loss_weights_by_k),
            ("bank_nmp_pass_loss_weights_by_k", self.bank_nmp_pass_loss_weights_by_k),
        ):
            if mapping is None:
                continue
            configured = set(mapping)
            if configured != pass_counts:
                raise ValueError(
                    f"{name} keys must exactly match sampled pass counts "
                    f"{sorted(pass_counts)}; got {sorted(configured)}"
                )
            for passes, weights in mapping.items():
                if len(weights) != passes:
                    raise ValueError(
                        f"{name}[{passes}] must contain exactly {passes} weights"
                    )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown experiment config fields: {unknown}")
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("pass_loss_weights", None)
        result.pop("pass_loss_weights_by_k", None)
        return result


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("experiment config must be a YAML mapping")
    return ExperimentConfig.from_dict(raw)
