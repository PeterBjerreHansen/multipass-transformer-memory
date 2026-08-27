#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import time
from typing import Any

import torch
import yaml

from tiny_mistral.device import resolve_device, synchronize
from tiny_mistral_mptt.data.packed_dataset import insert_memory_tokens
from tiny_mistral_mptt.model_factory import load_variant
from tiny_mistral_mptt.config import (
    MEMORY_ATTENTION_VARIANTS,
    MULTISCALE_MEMORY_ATTENTION_VARIANTS,
    canonical_memory_write_mode,
    canonical_variant_name,
)
from tiny_mistral_mptt.precision import PrecisionNotSupportedError, autocast_context
from tiny_mistral_mptt.training.phases import configure_phase


DEFAULT_MODEL_DIR = "checkpoints/TinyMistral-248M-v3"
WEIGHTS_BY_K = {
    1: [1.0],
    2: [0.25, 0.75],
    3: [0.05, 0.20, 0.75],
}


def _load_suite(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("efficiency suite must be a YAML mapping")
    defaults = raw.get("defaults", {})
    cases = raw.get("cases", [])
    if not isinstance(defaults, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("suite requires mapping defaults and a non-empty cases list")

    normalized: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be a mapping")
        item = {**defaults, **case}
        item.setdefault("grad_accum_steps", 1)
        for key in ("variant", "passes", "sequence_length", "batch_size", "grad_accum_steps"):
            if key not in item:
                raise ValueError(f"case {index} is missing {key!r}")
        normalized.append(item)
    return raw, normalized


def _apply_overrides(
    cases: list[dict[str, Any]],
    *,
    device: str | None,
    autocast_dtype: str | None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for case in cases:
        item = dict(case)
        if device is not None:
            item["device"] = device
        if autocast_dtype is not None:
            item["autocast_dtype"] = None if autocast_dtype == "none" else autocast_dtype
        result.append(item)
    return result


def _device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(
            {
                "device_name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "cuda_runtime": torch.version.cuda,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    elif device.type == "mps":
        info.update({"device_name": "Apple MPS", "bf16_supported": None})
    else:
        info.update({"device_name": platform.processor() or "CPU", "bf16_supported": None})
    return info


def _reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        empty = getattr(torch.mps, "empty_cache", None)
        if empty is not None:
            empty()


def _memory_metrics(device: torch.device) -> dict[str, int | None]:
    metrics: dict[str, int | None] = {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "allocated_bytes_after_measurement": None,
        "driver_allocated_bytes_after_measurement": None,
    }
    if device.type == "cuda":
        metrics["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        metrics["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    elif device.type == "mps" and hasattr(torch, "mps"):
        current = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        metrics["allocated_bytes_after_measurement"] = int(current()) if current else None
        metrics["driver_allocated_bytes_after_measurement"] = int(driver()) if driver else None
    return metrics


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        empty = getattr(torch.mps, "empty_cache", None)
        if empty is not None:
            empty()


def _module_parameter_dtypes(model: torch.nn.Module) -> list[str]:
    return sorted({str(parameter.dtype).removeprefix("torch.") for parameter in model.parameters()})


def _module_gradient_dtypes(model: torch.nn.Module) -> list[str]:
    return sorted(
        {
            str(parameter.grad.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.grad is not None
        }
    )


def _optimizer_state_dtypes(optimizer: torch.optim.Optimizer) -> list[str]:
    dtypes: set[str] = set()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                dtypes.add(str(value.dtype).removeprefix("torch."))
    return sorted(dtypes)


def _precision_error(text: str, autocast_dtype: str | None) -> bool:
    if autocast_dtype is None:
        return False
    lower = text.lower()
    mentions_precision = any(token in lower for token in ("bfloat16", "bf16", "autocast"))
    mentions_support = any(
        token in lower
        for token in ("not support", "unsupported", "not implemented", "unavailable")
    )
    return mentions_precision and mentions_support


def _run_case(case: dict[str, Any]) -> dict[str, Any]:
    device = resolve_device(str(case.get("device", "auto")))
    variant = str(case["variant"])
    implementation_variant = canonical_variant_name(variant)
    passes = int(case["passes"])
    sequence_length = int(case["sequence_length"])
    batch_size = int(case["batch_size"])
    grad_accum_steps = int(case.get("grad_accum_steps", 1))
    parameter_dtype = str(case.get("parameter_dtype", "float32"))
    autocast_dtype = case.get("autocast_dtype")
    if autocast_dtype in {"none", "null", ""}:
        autocast_dtype = None
    if autocast_dtype is not None:
        autocast_dtype = str(autocast_dtype)
    warmup_steps = int(case.get("warmup_steps", 2))
    measure_steps = int(case.get("measure_steps", 5))
    seed = int(case.get("seed", 1337))
    model_dir = str(case.get("model_dir", DEFAULT_MODEL_DIR))
    attention_backend = str(case.get("attention_backend", "auto"))
    memory_window = int(case.get("memory_window", 32))
    memory_write_mode = case.get("memory_write_mode")
    if memory_write_mode is not None:
        memory_write_mode = canonical_memory_write_mode(str(memory_write_mode))
    memory_write_stride = case.get("memory_write_stride")
    if memory_write_stride is not None:
        memory_write_stride = int(memory_write_stride)
    memory_token_visibility = case.get("memory_token_visibility")
    if memory_token_visibility is not None:
        memory_token_visibility = str(memory_token_visibility)
    memory_layers = case.get("memory_layers", "all")
    if memory_layers is None:
        memory_layers = "all"
    memory_position_encoding = str(case.get("memory_position_encoding", "rope"))
    memory_dense_window = int(case.get("memory_dense_window", 32))
    memory_sparse_window = int(case.get("memory_sparse_window", 32))
    memory_sparse_stride = int(case.get("memory_sparse_stride", 32))
    sparse_attention_stride = case.get("sparse_attention_stride")
    if sparse_attention_stride is not None:
        sparse_attention_stride = int(sparse_attention_stride)
    sparse_attention_window = case.get("sparse_attention_window")
    if sparse_attention_window is not None:
        sparse_attention_window = int(sparse_attention_window)
    sparse_attention_layers = case.get("sparse_attention_layers", "all")
    recirculation_source_layer = case.get("recirculation_source_layer")
    if recirculation_source_layer is not None:
        recirculation_source_layer = int(recirculation_source_layer)
    recirculation_destination_layer = case.get("recirculation_destination_layer")
    if recirculation_destination_layer is not None:
        recirculation_destination_layer = int(recirculation_destination_layer)
    recirculation_alpha = float(case.get("recirculation_alpha", 0.1))
    recirculation_mode = str(case.get("recirculation_mode", "fixed"))

    is_bank = variant in MEMORY_ATTENTION_VARIANTS
    if variant in MULTISCALE_MEMORY_ATTENTION_VARIANTS:
        if any(
            value is not None
            for value in (
                memory_write_mode,
                memory_write_stride,
                memory_token_visibility,
            )
        ):
            raise ValueError("Multiscale Memory Attention efficiency cases do not use memory_write_* fields")
        if min(memory_dense_window + memory_sparse_window, memory_sparse_stride) <= 0:
            raise ValueError("Multiscale Memory Attention efficiency cases require valid retention fields")
        memory_window = memory_dense_window + memory_sparse_window
    elif is_bank:
        if memory_write_mode not in {"dense", "periodic", "memory_token"}:
            raise ValueError("Memory Attention efficiency cases require memory_write_mode: dense|strided|memory_token")
        if memory_write_mode == "dense":
            if memory_write_stride is not None:
                raise ValueError("dense Memory Attention efficiency cases must not set memory_write_stride")
            if memory_token_visibility is not None:
                raise ValueError("memory_token_visibility applies only to memory_token mode")
        else:
            if memory_write_stride is None or memory_write_stride <= 0:
                raise ValueError(f"{memory_write_mode} Memory Attention requires positive memory_write_stride")
            if memory_write_mode == "periodic" and memory_token_visibility is not None:
                raise ValueError("memory_token_visibility applies only to memory_token mode")
            if memory_write_mode == "memory_token" and memory_token_visibility not in {"visible", "write_only"}:
                raise ValueError("memory-token Memory Attention requires memory_token_visibility: visible|write_only")
    elif any(value is not None for value in (memory_write_mode, memory_write_stride, memory_token_visibility)):
        raise ValueError("memory_* efficiency fields apply only to Memory Attention variants")

    if passes not in WEIGHTS_BY_K:
        raise ValueError("efficiency benchmark currently supports K=1,2,3")
    single_pass = implementation_variant in {"vanilla", "sparse_swa"}
    if single_pass and passes != 1:
        raise ValueError(f"{variant} efficiency cases require passes=1")
    if not single_pass and passes < 2:
        raise ValueError("multipass efficiency cases require passes>=2")
    if implementation_variant == "sparse_swa" and (
        sparse_attention_stride is None
        or sparse_attention_stride <= 0
        or sparse_attention_window is None
        or sparse_attention_window <= 0
    ):
        raise ValueError("sparse_swa efficiency cases require positive sparse attention fields")
    if min(sequence_length, batch_size, grad_accum_steps, measure_steps) <= 0 or warmup_steps < 0:
        raise ValueError(
            "sequence_length, batch_size, grad_accum_steps, and measure_steps must be positive"
        )
    if parameter_dtype != "float32" and autocast_dtype is not None:
        raise ValueError("autocast benchmark cases require FP32 parameter storage")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    result: dict[str, Any] = {
        "variant": variant,
        "passes": passes,
        "sequence_length": sequence_length,
        "linguistic_sequence_length": sequence_length,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "microbatch_tokens": batch_size * sequence_length,
        "optimizer_batch_tokens": batch_size * sequence_length * grad_accum_steps,
        "parameter_dtype": parameter_dtype,
        "autocast_dtype": autocast_dtype,
        "attention_backend": attention_backend,
        "memory_window": memory_window,
        "memory_write_mode": memory_write_mode,
        "memory_write_stride": memory_write_stride,
        "memory_token_visibility": memory_token_visibility,
        "memory_layers": memory_layers,
        "memory_position_encoding": memory_position_encoding,
        "memory_dense_window": memory_dense_window,
        "memory_sparse_window": memory_sparse_window,
        "memory_sparse_stride": memory_sparse_stride,
        "sparse_attention_stride": sparse_attention_stride,
        "sparse_attention_window": sparse_attention_window,
        "sparse_attention_layers": sparse_attention_layers,
        "recirculation_source_layer": recirculation_source_layer,
        "recirculation_destination_layer": recirculation_destination_layer,
        "recirculation_alpha": recirculation_alpha,
        "recirculation_mode": recirculation_mode,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "status": "running",
    }
    result.update(_device_info(device))

    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = load_variant(
            variant,
            model_dir,
            device=device,
            dtype=parameter_dtype,
            attention_backend=attention_backend,
            architecture_seed=4242,
            memory_window=memory_window,
            memory_write_mode=memory_write_mode,
            memory_write_stride=memory_write_stride,
            memory_token_visibility=memory_token_visibility,
            memory_layers=memory_layers,
            memory_position_encoding=memory_position_encoding,
            memory_dense_window=memory_dense_window,
            memory_sparse_window=memory_sparse_window,
            memory_sparse_stride=memory_sparse_stride,
            sparse_attention_stride=sparse_attention_stride,
            sparse_attention_window=sparse_attention_window,
            sparse_attention_layers=sparse_attention_layers,
            recirculation_source_layer=recirculation_source_layer,
            recirculation_destination_layer=recirculation_destination_layer,
            recirculation_alpha=recirculation_alpha,
            recirculation_mode=recirculation_mode,
        )
        configure_phase(model, "B")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-6, foreach=False)
        result["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        result["added_parameters_total"] = sum(
            parameter.numel() for parameter in model.added_parameters()
        )

        backbone = getattr(model, "backbone", None)
        vocab_size = int(getattr(getattr(backbone, "config", None), "vocab_size", 32005))
        ids = torch.randint(
            0,
            vocab_size,
            (batch_size, sequence_length),
            device=device,
            dtype=torch.long,
        )
        if memory_write_mode == "memory_token":
            assert memory_write_stride is not None
            ids = insert_memory_tokens(
                ids,
                memory_token_id=vocab_size,
                interval=memory_write_stride,
            )
        model_sequence_length = int(ids.shape[1])
        result["model_sequence_length"] = model_sequence_length
        result["microbatch_model_positions"] = batch_size * model_sequence_length
        result["optimizer_batch_model_positions"] = (
            batch_size * model_sequence_length * grad_accum_steps
        )
        weights = WEIGHTS_BY_K[passes]

        def step() -> tuple[float, float]:
            assert model is not None and optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(grad_accum_steps):
                with autocast_context(device, autocast_dtype):
                    output = model.compute_loss(
                        ids,
                        phase="B",
                        passes=passes,
                        loss_weights=weights,
                    )
                if not bool(torch.isfinite(output.loss).item()):
                    raise RuntimeError("non-finite loss")
                (output.loss / grad_accum_steps).backward()
                accumulated_loss += float(output.loss.detach().cpu())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            synchronize(device)
            return accumulated_loss / grad_accum_steps, float(grad_norm.detach().cpu())

        for _ in range(warmup_steps):
            step()

        _reset_memory(device)
        synchronize(device)
        started = time.perf_counter()
        losses: list[float] = []
        grad_norms: list[float] = []
        for _ in range(measure_steps):
            loss, grad_norm = step()
            losses.append(loss)
            grad_norms.append(grad_norm)
        elapsed = max(time.perf_counter() - started, 1e-12)

        unique_tokens = batch_size * sequence_length * grad_accum_steps * measure_steps
        model_positions = batch_size * model_sequence_length * grad_accum_steps * measure_steps
        pass_positions = model_positions * passes
        unique_tokens_per_second = unique_tokens / elapsed
        result.update(
            {
                "status": "ok",
                "elapsed_seconds": elapsed,
                "milliseconds_per_step": elapsed * 1000.0 / measure_steps,
                "milliseconds_per_optimizer_step": elapsed * 1000.0 / measure_steps,
                "optimizer_steps_per_second": measure_steps / elapsed,
                "microbatches_per_second": (measure_steps * grad_accum_steps) / elapsed,
                "unique_tokens_per_second": unique_tokens_per_second,
                "model_positions_per_second": model_positions / elapsed,
                "pass_positions_per_second": pass_positions / elapsed,
                "estimated_hours_per_100m_unique_tokens": 100_000_000.0
                / unique_tokens_per_second
                / 3600.0,
                "mean_loss": sum(losses) / len(losses),
                "mean_grad_norm": sum(grad_norms) / len(grad_norms),
                "actual_parameter_dtypes": _module_parameter_dtypes(model),
                "gradient_dtypes": _module_gradient_dtypes(model),
                "optimizer_state_dtypes": _optimizer_state_dtypes(optimizer),
                **_memory_metrics(device),
            }
        )
    except PrecisionNotSupportedError as exc:
        result.update({"status": "unsupported", "error": str(exc)})
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "oom", "error": str(exc)})
    except (RuntimeError, NotImplementedError) as exc:
        text = str(exc)
        lower = text.lower()
        if "out of memory" in lower:
            result.update({"status": "oom", "error": text})
        elif _precision_error(text, autocast_dtype):
            result.update({"status": "unsupported", "error": text})
        else:
            result.update({"status": "error", "error": text})
    finally:
        del optimizer
        del model
        _cleanup(device)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure end-to-end TinyMistral training efficiency on MPS or CUDA."
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda", "auto"),
        default=None,
        help="override the suite device for every case",
    )
    parser.add_argument(
        "--autocast-dtype",
        choices=("none", "bfloat16"),
        default=None,
        help="override suite autocast for every case",
    )
    args = parser.parse_args()

    suite, cases = _load_suite(args.suite)
    cases = _apply_overrides(
        cases,
        device=args.device,
        autocast_dtype=args.autocast_dtype,
    )

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] {case['variant']} K={case['passes']} "
            f"T={case['sequence_length']} B={case['batch_size']} "
            f"A={case.get('grad_accum_steps', 1)} "
            f"device={case.get('device', 'auto')} autocast={case.get('autocast_dtype')}"
        )
        row = _run_case(case)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    document = {
        "suite": suite.get("name", Path(args.suite).stem),
        "suite_path": str(args.suite),
        "overrides": {
            "device": args.device,
            "autocast_dtype": args.autocast_dtype,
        },
        "results": rows,
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
