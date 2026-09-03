from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch

from tiny_mistral.device import mps_available


SUPPORTED_AUTOCAST_DTYPES = {"bfloat16"}


class PrecisionNotSupportedError(RuntimeError):
    """Raised when a requested reduced-precision compute mode is unavailable."""


def autocast_context(
    device: torch.device | str,
    dtype: str | None,
) -> AbstractContextManager:
    """Return the configured training/evaluation autocast context.

    Learned parameters remain FP32 when this is used through ``ExperimentConfig``.
    BF16 autocast is supported on CUDA and is exposed on MPS when the installed
    PyTorch/macOS stack accepts it. MPS support is intentionally capability-
    checked at runtime because it varies more across host versions.
    """
    if dtype is None:
        return nullcontext()
    if dtype not in SUPPORTED_AUTOCAST_DTYPES:
        raise ValueError(
            f"unsupported autocast dtype {dtype!r}; expected one of "
            f"{sorted(SUPPORTED_AUTOCAST_DTYPES)}"
        )

    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise PrecisionNotSupportedError("CUDA autocast requested but CUDA is unavailable")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise PrecisionNotSupportedError(
                "CUDA BF16 autocast requested but this GPU does not support BF16"
            )
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    if resolved.type == "mps":
        if not mps_available():
            raise PrecisionNotSupportedError("MPS autocast requested but MPS is unavailable")
        try:
            return torch.autocast(device_type="mps", dtype=torch.bfloat16)
        except (RuntimeError, TypeError) as exc:
            raise PrecisionNotSupportedError(
                "MPS BF16 autocast is unavailable on this PyTorch/macOS stack"
            ) from exc

    raise PrecisionNotSupportedError(
        f"BF16 autocast is supported only on CUDA or MPS; resolved device={resolved.type}"
    )
