"""Resolve ordinary evaluation parameters, without named policy profiles."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

from ..config import ExperimentConfig
from ..variants.multipass import MultiPassVariant


@dataclass(frozen=True)
class EvaluationSettings:
    passes: int
    forward_mode: str
    prefill_passes: int
    decode_mode: str
    autocast_dtype: str | None


def resolve_evaluation_settings(
    config: ExperimentConfig, model, *, passes: int | None = None,
    forward_mode: str | None = None, prefill_passes: int | None = None,
    decode_mode: str | None = None, autocast_dtype: str = "config",
) -> EvaluationSettings:
    """Explicit overrides win; otherwise use experiment defaults.

    The fallback is feedback for cached-feedback models, standard otherwise.
    It never depends on K. Precision overrides do not mutate the training config
    used for checkpoint compatibility checks.
    """
    forward = config.validation_forward if forward_mode is None else forward_mode
    bos_feedback = forward == "feedback"
    depth = (1 if bos_feedback else config.eval_passes) if passes is None else passes
    prefill = config.eval_prefill_passes if prefill_passes is None else prefill_passes
    if prefill is None:
        prefill = config.eval_passes
    if bos_feedback and prefill_passes is None:
        prefill = 1
    feedback_supported = bool(getattr(model, "supports_cached_feedback", False))
    mode = config.eval_decode_mode if decode_mode is None else decode_mode
    if mode is None:
        mode = "feedback" if feedback_supported else "standard"
    if bos_feedback and decode_mode is None:
        mode = "feedback"
    if depth < 1 or prefill < 1:
        raise ValueError("passes and prefill_passes must be positive")
    if forward not in {"parallel_multipass", "feedback"}:
        raise ValueError("forward_mode must be parallel_multipass or feedback")
    if bos_feedback and (depth != 1 or prefill != 1 or mode != "feedback"):
        raise ValueError("BOS-only feedback NLL requires passes=1, prefill_passes=1 and feedback decoding")
    if mode not in {"standard", "feedback"}:
        raise ValueError("decode_mode must be standard or feedback")
    if mode == "feedback" and not feedback_supported:
        raise ValueError("loaded model does not implement feedback decoding")
    if not isinstance(model, MultiPassVariant) and (depth != 1 or prefill != 1):
        raise ValueError("single-pass models require passes=1 and prefill_passes=1")
    if autocast_dtype not in {"config", "float32", "bfloat16"}:
        raise ValueError("autocast_dtype override must be config, float32 or bfloat16")
    precision = config.autocast_dtype if autocast_dtype == "config" else (
        None if autocast_dtype == "float32" else autocast_dtype
    )
    return EvaluationSettings(depth, forward, prefill, mode, precision)


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--autocast-dtype", choices=("config", "float32", "bfloat16"), default="config",
        help="default: experiment autocast_dtype; float32 explicitly disables autocast",
    )
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "auto"), default=None,
        help="override the experiment device without changing checkpoint identity",
    )
