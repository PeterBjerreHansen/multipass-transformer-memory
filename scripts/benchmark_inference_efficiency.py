#!/usr/bin/env python
"""Measure the GPU cost of the feedback-evaluation paths.

This benchmark is intentionally synthetic: it measures model and evaluator
compute without requiring a copy of the held-out data on the benchmark VM.
Token IDs are ordinary vocabulary IDs, so no memory-token/control-token path is
implicitly included.  The output reports both raw timing curves and derived
costs for the routine 64-block check and the roughly 2M-token validation split.

The diagnostic loop mirrors ``evaluate_feedback_continuation`` closely,
including its per-token loss, hidden-drift, cosine, and host-transfer work.
That row estimates the combined continuation diagnostic; the feedback-only row
isolates model-side cached recurrence. Neither measures full-block BOS-only NLL.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import platform
import time
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from tiny_mistral_mptt.config import load_experiment_config
from tiny_mistral_mptt.inference import (
    exact_decode_step,
    prefill_exact_k_pass,
    prefill_live_feedback,
    live_feedback_from_exact,
    live_feedback_decode_step,
)
from tiny_mistral_mptt.model_factory import load_variant_from_config
from tiny_mistral_mptt.precision import autocast_context
from tiny_mistral_mptt.variants.multipass import MultiPassVariant


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    "benchmarks/development/frozen_backbone_comparison/no_memory_adapter_one_site_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/recurrent_projected_residual_multipass_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/recurrent_recirculation_multipass_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/dense_memory_attention_one_site_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/no_memory_adapter_two_site_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/recurrent_projected_residual_two_site_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/recurrent_recirculation_two_site_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/dense_memory_attention_multipass_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/strided_memory_attention_one_site_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/dense_and_strided_memory_attention_one_site_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/strided_memory_attention_multipass_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/dense_and_strided_memory_attention_multipass_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/strided_memory_attention_stride8_two_site_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/strided_memory_attention_stride16_two_site_100m.yaml",
    "benchmarks/development/frozen_backbone_comparison/strided_memory_attention_stride64_two_site_100m.yaml",
)
DEFAULT_PROMPT_LENGTHS = (1, 256)
DEFAULT_HORIZONS = (1, 16, 64, 256, 512, 1024, 2047)


@dataclass(frozen=True)
class Timing:
    samples_ms: tuple[float, ...]
    median_ms: float
    min_ms: float
    max_ms: float


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _timing(samples: Iterable[float]) -> Timing:
    values = tuple(float(value) for value in samples)
    if not values:
        raise ValueError("timing requires at least one sample")
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle]
    if len(ordered) % 2 == 0:
        median = (ordered[middle - 1] + ordered[middle]) / 2.0
    return Timing(
        samples_ms=values,
        median_ms=median,
        min_ms=min(values),
        max_ms=max(values),
    )


def _measure(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    repeats: int,
    warmups: int,
) -> Timing:
    for _ in range(warmups):
        result = fn()
        del result
        _sync(device)

    samples: list[float] = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        result = fn()
        _sync(device)
        samples.append((time.perf_counter() - started) * 1000.0)
        del result
    return _timing(samples)


def _make_tokens(
    *,
    device: torch.device,
    vocab_size: int,
    length: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    # Avoid special/control IDs while keeping the workload deterministic.
    high = max(2, min(int(vocab_size), 32000))
    return torch.randint(
        1,
        high,
        (1, length),
        generator=generator,
        dtype=torch.long,
    ).to(device)


def _precision_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision == "bfloat16":
        return autocast_context(device, "bfloat16")
    raise ValueError(f"unknown precision {precision!r}")


def _measure_prefill(
    model: MultiPassVariant,
    prompt: torch.Tensor,
    *,
    passes: int,
    decode_mode: str,
    precision: str,
    device: torch.device,
    repeats: int,
    warmups: int,
) -> Timing:
    def run():
        with _precision_context(device, precision):
            return prefill_live_feedback(
                model,
                prompt,
                passes=passes,
                decode_mode=decode_mode,
            )

    return _measure(run, device=device, repeats=repeats, warmups=warmups)


def _decode_loop(
    model: MultiPassVariant,
    state: Any,
    tokens: torch.Tensor,
    *,
    step: Callable[[MultiPassVariant, Any, torch.Tensor], Any],
    precision: str,
    device: torch.device,
    checkpoints: tuple[int, ...],
    include_diagnostic_metrics: bool = False,
) -> dict[int, float]:
    checkpoint_set = set(checkpoints)
    elapsed: dict[int, float] = {}
    started = time.perf_counter()
    if not checkpoints:
        raise ValueError("decode timing requires at least one checkpoint")
    measured_tokens = max(checkpoints)
    if measured_tokens > tokens.shape[1]:
        raise ValueError("decode checkpoint exceeds continuation length")
    for offset in range(measured_tokens):
        target = tokens[:, offset : offset + 1]
        if include_diagnostic_metrics:
            # Keep this in lockstep with evaluate_feedback_continuation. The
            # .cpu() calls deliberately retain the host-synchronization cost of
            # the current evaluator rather than hiding it from the benchmark.
            valid = ~model.control_token_mask(target)[:, 0]
            exact_state, recurrent_state, vanilla_state = state
            for current in (
                exact_state.next_token_logits,
                recurrent_state.next_token_logits,
                vanilla_state.next_token_logits,
            ):
                _ = F.cross_entropy(
                    current[valid].float(),
                    target[valid, 0],
                    reduction="sum",
                ).detach().cpu()
        state = step(model, state, target)
        if include_diagnostic_metrics:
            exact_state, recurrent_state, vanilla_state = state
            delta = recurrent_state.last_hidden.float() - exact_state.last_hidden.float()
            _ = delta.square().sum().cpu()
            cosine = F.cosine_similarity(
                recurrent_state.last_hidden.float(),
                exact_state.last_hidden.float(),
                dim=-1,
            )
            _ = cosine.sum().cpu()
        if offset + 1 in checkpoint_set:
            _sync(device)
            elapsed[offset + 1] = (time.perf_counter() - started) * 1000.0
    return elapsed


def _standard_step(model: MultiPassVariant, state: Any, token: torch.Tensor):
    return live_feedback_decode_step(model, state, token)


def _exact_step(model: MultiPassVariant, state: Any, token: torch.Tensor):
    return exact_decode_step(model, state, token)


def _diagnostic_step(model: MultiPassVariant, state: Any, token: torch.Tensor):
    exact, recurrent, vanilla = state
    return (
        exact_decode_step(model, exact, token),
        live_feedback_decode_step(model, recurrent, token),
        exact_decode_step(model, vanilla, token),
    )


def _measure_decode_curve(
    make_state: Callable[[], Any],
    model: MultiPassVariant,
    continuation: torch.Tensor,
    *,
    step: Callable[[MultiPassVariant, Any, torch.Tensor], Any],
    precision: str,
    device: torch.device,
    checkpoints: tuple[int, ...],
    repeats: int,
    warmups: int,
    include_diagnostic_metrics: bool = False,
) -> dict[str, Any]:
    def run() -> dict[int, float]:
        with _precision_context(device, precision):
            return _decode_loop(
                model,
                make_state(),
                continuation,
                step=step,
                precision=precision,
                device=device,
                checkpoints=checkpoints,
                include_diagnostic_metrics=include_diagnostic_metrics,
            )

    for _ in range(warmups):
        result = run()
        del result
        _sync(device)

    samples: dict[int, list[float]] = {point: [] for point in checkpoints}
    for _ in range(repeats):
        result = run()
        for point, value in result.items():
            samples[point].append(value)
        del result
        _sync(device)
    return {
        str(point): asdict(_timing(values))
        for point, values in samples.items()
    }


def _row(
    *,
    variant: str,
    precision: str,
    operation: str,
    mode: str,
    prompt_tokens: int | None,
    continuation_tokens: int | None,
    timings: Timing | dict[str, Any],
    memory: dict[str, int | None],
) -> dict[str, Any]:
    payload = asdict(timings) if isinstance(timings, Timing) else timings
    return {
        "variant": variant,
        "precision": precision,
        "operation": operation,
        "mode": mode,
        "prompt_tokens": prompt_tokens,
        "continuation_tokens": continuation_tokens,
        "timing": payload,
        "memory": memory,
    }


def _derived_costs(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    precision: str,
    sequence_length: int,
    validation_tokens: int,
    routine_blocks: int,
    arm: str | None = None,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["variant"] == variant and row["precision"] == precision
        and (arm is None or row.get("arm") == arm)
    ]
    if arm is None and len({row.get("arm") for row in selected}) > 1:
        raise ValueError("multiple arms share this variant; select an explicit arm")

    def median(
        operation: str,
        mode: str,
        prompt: int | None = None,
        *,
        extrapolate_to: int | None = None,
    ) -> tuple[float, int | None]:
        for row in selected:
            if (
                row["operation"] == operation
                and row["mode"] == mode
                and (prompt is None or row["prompt_tokens"] == prompt)
            ):
                value = row["timing"]
                if isinstance(value, dict) and "median_ms" in value:
                    return float(value["median_ms"]), None
                target = sequence_length - 1 if extrapolate_to is None else extrapolate_to
                full = value.get(str(target))
                if full is not None:
                    return float(full["median_ms"]), None
                if extrapolate_to is not None and mode in {
                    "feedback_k4",
                    "full_diagnostic_k4",
                }:
                    measured = max(int(key) for key in value)
                    return (
                        float(value[str(measured)]["median_ms"])
                        * extrapolate_to
                        / measured,
                        measured,
                    )
                raise KeyError(f"missing horizon {target} for {mode}")
        raise KeyError(f"missing timing row {operation}/{mode}/{prompt}")

    block_count = (validation_tokens + sequence_length - 1) // sequence_length
    k4_validation_ms, _ = median("full_sequence", "k4")
    k1_prefill_ms, _ = median("prefill", "k1_standard", prompt=1)
    k4_prefill_ms, _ = median("prefill", "k4_feedback", prompt=1)
    diagnostic_decode_ms, diagnostic_measured_horizon = median(
        "decode_curve",
        "full_diagnostic_k4",
        prompt=1,
        extrapolate_to=sequence_length - 1,
    )
    feedback_decode_ms, feedback_measured_horizon = median(
        "decode_curve",
        "feedback_k4",
        prompt=1,
        extrapolate_to=sequence_length - 1,
    )
    total_feedback_diagnostic_ms = (
        k1_prefill_ms + k4_prefill_ms + diagnostic_decode_ms
    )
    return {
        "validation_blocks": block_count,
        "routine_validation_blocks": routine_blocks,
        "k4_validation_seconds_per_block": k4_validation_ms / 1000.0,
        "routine_k4_validation_seconds": k4_validation_ms * routine_blocks / 1000.0,
        "full_k4_validation_seconds": k4_validation_ms * block_count / 1000.0,
        "feedback_only_seconds_per_block": (
            k4_prefill_ms + feedback_decode_ms
        ) / 1000.0,
        "feedback_decode_extrapolated": feedback_measured_horizon is not None,
        "feedback_decode_measured_horizon": feedback_measured_horizon,
        "full_diagnostic_seconds_per_block": total_feedback_diagnostic_ms / 1000.0,
        "routine_feedback_diagnostic_seconds": (
            total_feedback_diagnostic_ms * routine_blocks / 1000.0
        ),
        "full_feedback_diagnostic_seconds": (
            total_feedback_diagnostic_ms * block_count / 1000.0
        ),
        "full_diagnostic_decode_extrapolated": diagnostic_measured_horizon is not None,
        "full_diagnostic_decode_measured_horizon": diagnostic_measured_horizon,
        "feedback_diagnostic_over_k4_validation": (
            total_feedback_diagnostic_ms / k4_validation_ms
        ),
    }


def _device_info(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "device_name": props.name,
        "total_memory_bytes": int(props.total_memory),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "python": platform.python_version(),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def _run_variant(
    config_path: Path,
    *,
    device: torch.device,
    precisions: tuple[str, ...],
    prompt_lengths: tuple[int, ...],
    horizons: tuple[int, ...],
    sequence_length: int,
    diagnostic_horizon: int,
    repeats: int,
    warmups: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = load_experiment_config(config_path)
    started = time.perf_counter()
    model = load_variant_from_config(cfg, device=device)
    load_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(model, MultiPassVariant):
        raise TypeError(f"{cfg.variant} is not a MultiPassVariant")
    model.eval()
    vocab_size = int(model.config.vocab_size)
    rows: list[dict[str, Any]] = []

    for precision in precisions:
        for prompt_length in prompt_lengths:
            if prompt_length < 1 or prompt_length >= sequence_length:
                raise ValueError("prompt lengths must lie in [1, sequence_length)")
            prompt = _make_tokens(
                device=device,
                vocab_size=vocab_size,
                length=prompt_length,
                seed=seed + prompt_length,
            )
            max_continuation = sequence_length - prompt_length
            points = tuple(
                point for point in horizons if 1 <= point <= max_continuation
            )
            continuation = _make_tokens(
                device=device,
                vocab_size=vocab_size,
                length=max_continuation,
                seed=seed + 10000 + prompt_length,
            )

            for passes, mode, label in (
                (1, "standard", "k1_standard"),
                (4, "feedback", "k4_feedback"),
            ):
                torch.cuda.reset_peak_memory_stats(device)
                timing = _measure_prefill(
                    model,
                    prompt,
                    passes=passes,
                    decode_mode=mode,
                    precision=precision,
                    device=device,
                    repeats=repeats,
                    warmups=warmups,
                )
                rows.append(
                    _row(
                        variant=cfg.variant,
                        precision=precision,
                        operation="prefill",
                        mode=label,
                        prompt_tokens=prompt_length,
                        continuation_tokens=None,
                        timings=timing,
                        memory=_memory(device),
                    )
                )

            def make_standard():
                with _precision_context(device, precision):
                    return prefill_live_feedback(
                        model, prompt, passes=1, decode_mode="standard"
                    )

            def make_feedback():
                with _precision_context(device, precision):
                    return prefill_live_feedback(
                        model, prompt, passes=4, decode_mode="feedback"
                    )

            def make_exact():
                with _precision_context(device, precision):
                    return prefill_exact_k_pass(model, prompt, passes=4)

            for mode, make_state, step in (
                ("standard_k1", make_standard, _standard_step),
                ("feedback_k4", make_feedback, _standard_step),
                ("exact_k4", make_exact, _exact_step),
            ):
                torch.cuda.reset_peak_memory_stats(device)
                curve = _measure_decode_curve(
                    make_state,
                    model,
                    continuation,
                    step=step,
                    precision=precision,
                    device=device,
                    checkpoints=points,
                    repeats=repeats,
                    warmups=warmups,
                )
                rows.append(
                    _row(
                        variant=cfg.variant,
                        precision=precision,
                        operation="decode_curve",
                        mode=mode,
                        prompt_tokens=prompt_length,
                        continuation_tokens=max_continuation,
                        timings=curve,
                        memory=_memory(device),
                    )
                )

            def make_diagnostic():
                with _precision_context(device, precision):
                    exact = prefill_exact_k_pass(model, prompt, passes=4)
                    recurrent = live_feedback_from_exact(exact, decode_mode="feedback")
                    vanilla = prefill_exact_k_pass(model, prompt, passes=1)
                    return exact, recurrent, vanilla

            torch.cuda.reset_peak_memory_stats(device)
            diagnostic_length = min(max_continuation, diagnostic_horizon)
            diagnostic = _measure_decode_curve(
                make_diagnostic,
                model,
                continuation[:, :diagnostic_length],
                step=_diagnostic_step,
                precision=precision,
                device=device,
                checkpoints=tuple(point for point in points if point <= diagnostic_length),
                repeats=repeats,
                warmups=warmups,
                include_diagnostic_metrics=True,
            )
            rows.append(
                _row(
                    variant=cfg.variant,
                    precision=precision,
                    operation="decode_curve",
                    mode="full_diagnostic_k4",
                    prompt_tokens=prompt_length,
                    continuation_tokens=diagnostic_length,
                    timings=diagnostic,
                    memory=_memory(device),
                )
            )

        full_input = _make_tokens(
            device=device,
            vocab_size=vocab_size,
            length=sequence_length,
            seed=seed + 20000,
        )
        for passes, label in ((1, "k1"), (4, "k4")):
            def run_full() -> Any:
                with _precision_context(device, precision):
                    return model.compute_passes(full_input, passes=passes, phase="B")

            torch.cuda.reset_peak_memory_stats(device)
            timing = _measure(
                run_full,
                device=device,
                repeats=repeats,
                warmups=warmups,
            )
            rows.append(
                _row(
                    variant=cfg.variant,
                    precision=precision,
                    operation="full_sequence",
                    mode=label,
                    prompt_tokens=None,
                    continuation_tokens=sequence_length,
                    timings=timing,
                    memory=_memory(device),
                )
            )

    for row in rows:
        row["arm"] = config_path.stem
    metadata = {
        "arm": config_path.stem,
        "config": str(config_path),
        "variant": cfg.variant,
        "recurrent_merger": cfg.recurrent_merger,
        "recurrent_layers": cfg.recurrent_layers,
        "memory_pattern": cfg.memory_pattern,
        "memory_layers": cfg.memory_layers,
        "model_load_ms": load_ms,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark exact and feedback inference evaluation on CUDA."
    )
    parser.add_argument("--config", action="append", dest="configs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bfloat16", "both"),
        default="both",
    )
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=list(DEFAULT_PROMPT_LENGTHS))
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--diagnostic-horizon",
        type=int,
        default=512,
        help="measure the host-synchronizing diagnostic to this horizon and extrapolate the full-block cost",
    )
    parser.add_argument("--validation-tokens", type=int, default=2_000_000)
    parser.add_argument("--routine-blocks", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.repeats < 1 or args.warmups < 0:
        raise SystemExit("repeats must be positive and warmups must be non-negative")
    if args.sequence_length < 2 or args.validation_tokens < 1 or args.routine_blocks < 1:
        raise SystemExit("sequence length, validation tokens, and routine blocks must be positive")
    if args.diagnostic_horizon < 1:
        raise SystemExit("diagnostic horizon must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this benchmark requires an available CUDA device")
    if args.precision in {"bfloat16", "both"} and not torch.cuda.is_bf16_supported():
        raise SystemExit("requested BF16 benchmark but the CUDA device does not support BF16")
    precisions = ("fp32", "bfloat16") if args.precision == "both" else (args.precision,)
    configs = tuple(Path(path) for path in (args.configs or DEFAULT_CONFIGS))
    configs = tuple(
        path if path.is_absolute() else REPOSITORY_ROOT / path for path in configs
    )
    for path in configs:
        if not path.is_file():
            raise SystemExit(f"config does not exist: {path}")
    if len({path.stem for path in configs}) != len(configs):
        raise SystemExit("benchmark configs must have distinct arm names (filename stems)")

    all_rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for config in configs:
        rows, info = _run_variant(
            config,
            device=device,
            precisions=precisions,
            prompt_lengths=tuple(args.prompt_lengths),
            horizons=tuple(sorted(set(args.horizons))),
            sequence_length=args.sequence_length,
            diagnostic_horizon=args.diagnostic_horizon,
            repeats=args.repeats,
            warmups=args.warmups,
            seed=args.seed,
        )
        all_rows.extend(rows)
        metadata.append(info)
        print(f"completed {info['arm']}", flush=True)

    derived = [
        {
            "arm": info["arm"],
            "variant": info["variant"],
            "precision": precision,
            "costs": _derived_costs(
                all_rows,
                variant=info["variant"],
                arm=info["arm"],
                precision=precision,
                sequence_length=args.sequence_length,
                validation_tokens=args.validation_tokens,
                routine_blocks=args.routine_blocks,
            ),
        }
        for info in metadata
        for precision in precisions
    ]
    document = {
        "schema_version": 2,
        "benchmark": "inference_efficiency_feedback_evaluation",
        "synthetic_inputs": True,
        "sequence_length": args.sequence_length,
        "validation_tokens": args.validation_tokens,
        "routine_blocks": args.routine_blocks,
        "prompt_lengths": list(args.prompt_lengths),
        "horizons": list(sorted(set(args.horizons))),
        "diagnostic_horizon": args.diagnostic_horizon,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "device_info": _device_info(device),
        "model_metadata": metadata,
        "rows": all_rows,
        "derived_costs": derived,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
