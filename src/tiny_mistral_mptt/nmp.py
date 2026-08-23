from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_mistral.modeling import MistralRMSNorm


NMP_TARGET_NORMALIZATIONS = {"none", "rms"}


def normalize_nmp_target(
    states: torch.Tensor,
    *,
    normalization: str,
    eps: float,
) -> torch.Tensor:
    """Apply the configured parameter-free normalization to NMP targets.

    ``rms`` deliberately matches the variance calculation used by the
    model's RMSNorm, but omits its learned feature-wise gain.  Target
    normalization must not add trainable parameters or create a gradient path
    through the future memory.  ``none`` is the exact stored representation.
    """

    if normalization not in NMP_TARGET_NORMALIZATIONS:
        raise ValueError(
            "NMP target normalization must be one of "
            f"{sorted(NMP_TARGET_NORMALIZATIONS)}"
        )
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("NMP target normalization eps must be finite and positive")
    if not states.is_floating_point():
        raise ValueError("NMP targets must have a floating-point dtype")
    if normalization == "none":
        return states
    values = states.to(torch.float32)
    variance = values.square().mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + float(eps))
    return normalized.to(states.dtype)


class LatentPredictionHead(nn.Module):
    """Predict a future memory from one current top-layer hidden state.

    The deliberately narrow ``forward(hidden_states)`` API is part of the
    causality contract: token embeddings and future observations cannot enter
    this prediction path.  The final projection starts at zero so enabling NMP
    does not inject an arbitrary latent prediction into the first update.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        projection_factor: float,
        rms_norm_eps: float,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(projection_factor)) or projection_factor <= 0:
            raise ValueError("projection_factor must be finite and positive")
        width = max(128, int(math.ceil(hidden_size * projection_factor / 128.0)) * 128)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.input_norm = MistralRMSNorm(hidden_size, eps=float(rms_norm_eps))
            self.hidden_1 = nn.Linear(hidden_size, width)
            self.hidden_2 = nn.Linear(width, width)
            self.output = nn.Linear(width, hidden_size)
            nn.init.xavier_uniform_(self.hidden_1.weight)
            nn.init.zeros_(self.hidden_1.bias)
            nn.init.xavier_uniform_(self.hidden_2.weight)
            nn.init.zeros_(self.hidden_2.bias)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("NMP predictor input must be [B,T,D]")
        hidden = F.gelu(self.hidden_1(self.input_norm(hidden_states)))
        hidden = F.gelu(self.hidden_2(hidden))
        return self.output(hidden)


def next_strict_true_indices(mask: torch.Tensor) -> torch.Tensor:
    """Index of the first true position strictly to the right, or ``T``.

    This device-side suffix scan is shared by recurrent and bank NMP.  The
    one-position shift is essential: a write at query position ``t`` can never
    become that query's target.
    """

    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError("mask must be bool [B,T]")
    batch, length = mask.shape
    positions = torch.arange(length, device=mask.device, dtype=torch.long)
    sentinel = torch.full((batch, length), length, device=mask.device, dtype=torch.long)
    candidates = torch.where(mask, positions[None, :].expand(batch, -1), sentinel)
    suffix_min = torch.flip(
        torch.cummin(torch.flip(candidates, dims=(1,)), dim=1).values,
        dims=(1,),
    )
    return torch.cat(
        (
            suffix_min[:, 1:],
            torch.full((batch, 1), length, device=mask.device, dtype=torch.long),
        ),
        dim=1,
    )


def _gather_states(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    safe = indices.clamp(max=states.shape[1] - 1)
    return states.gather(1, safe[:, :, None].expand(-1, -1, states.shape[-1]))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.float()
    weights = mask.to(dtype=values.dtype)
    count = weights.sum()
    mean = (values * weights).sum() / count.clamp_min(1.0)
    return torch.where(count.gt(0), mean, values.sum() * 0.0)


def target_diagnostics(
    targets: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target RMS and mean feature standard deviation.

    Empty sparse batches are valid and return differentiable zeros.
    """

    if targets.ndim != 3 or valid.shape != targets.shape[:2]:
        raise ValueError("targets must be [B,T,D] and valid must be [B,T]")
    values = targets.float()
    weights = valid.to(dtype=values.dtype)
    count = weights.sum()
    denominator = count.clamp_min(1.0) * values.shape[-1]
    rms = (values.square() * weights[..., None]).sum().div(denominator).sqrt()
    mean = (values * weights[..., None]).sum(dim=(0, 1)).div(count.clamp_min(1.0))
    variance = (
        (values - mean).square() * weights[..., None]
    ).sum(dim=(0, 1)).div(count.clamp_min(1.0))
    feature_std = variance.clamp_min(0.0).sqrt().mean()
    has_target = count.gt(0)
    zero = values.sum() * 0.0
    return torch.where(has_target, rms, zero), torch.where(has_target, feature_std, zero)


def _error_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return error RMS and fraction beyond Smooth-L1 beta=1."""

    error = predictions.float() - targets.float()
    mask = valid[..., None]
    denominator = valid.to(dtype=error.dtype).sum().clamp_min(1.0) * error.shape[-1]
    error_rms = (error.square() * mask).sum().div(denominator).sqrt()
    linear_fraction = (
        error.abs().ge(1.0).to(dtype=error.dtype) * mask
    ).sum().div(denominator)
    has_target = valid.any()
    zero = error.sum() * 0.0
    return torch.where(has_target, error_rms, zero), torch.where(
        has_target, linear_fraction, zero
    )


@dataclass(frozen=True)
class RecurrentNMPAlignment:
    """Shared recurrent NMP target alignment for every computed pass."""

    targets: torch.Tensor
    valid: torch.Tensor
    target_rms: torch.Tensor
    target_feature_std: torch.Tensor


@dataclass(frozen=True)
class BankNMPAlignment:
    """Shared Bank NMP event alignment for every computed pass."""

    targets: torch.Tensor
    valid: torch.Tensor
    safe_write: torch.Tensor
    event_counts: torch.Tensor
    present_events: torch.Tensor
    has_target_examples: torch.Tensor
    distance_masks: dict[str, torch.Tensor]
    target_rms: torch.Tensor
    target_feature_std: torch.Tensor


def prepare_recurrent_nmp_alignment(
    final_targets: torch.Tensor,
    *,
    ordinary_mask: torch.Tensor,
) -> RecurrentNMPAlignment:
    """Build recurrent next-token indices and targets once per batch."""

    if final_targets.ndim != 3:
        raise ValueError("recurrent NMP targets must be [B,T,D]")
    if ordinary_mask.shape != final_targets.shape[:2] or ordinary_mask.dtype != torch.bool:
        raise ValueError("ordinary mask shape/dtype differs from recurrent NMP targets")
    next_index = next_strict_true_indices(ordinary_mask)
    valid = ordinary_mask & next_index.lt(final_targets.shape[1])
    targets = _gather_states(final_targets, next_index).detach()
    target_rms, target_feature_std = target_diagnostics(targets, valid)
    return RecurrentNMPAlignment(targets, valid, target_rms, target_feature_std)


def prepare_bank_nmp_alignment(
    final_written_states: torch.Tensor,
    *,
    ordinary_mask: torch.Tensor,
    write_mask: torch.Tensor,
    sequence_positions: torch.Tensor,
) -> BankNMPAlignment:
    """Build future-write events, targets, and distances once per batch."""

    if final_written_states.ndim != 3:
        raise ValueError("bank NMP targets must be [B,T,D]")
    if (
        ordinary_mask.shape != final_written_states.shape[:2]
        or write_mask.shape != ordinary_mask.shape
        or ordinary_mask.dtype != torch.bool
        or write_mask.dtype != torch.bool
    ):
        raise ValueError("bank NMP masks must be bool [B,T] with matching shapes")
    if sequence_positions.shape != ordinary_mask.shape:
        raise ValueError("bank NMP sequence positions must match [B,T]")

    batch, length, _ = final_written_states.shape
    next_write = next_strict_true_indices(write_mask)
    valid = ordinary_mask & next_write.lt(length)
    safe_write = next_write.clamp(max=length - 1)
    targets = _gather_states(final_written_states, next_write).detach()
    event_counts = torch.zeros(
        (batch, length), device=valid.device, dtype=torch.float32
    ).scatter_add(1, safe_write, valid.float())
    present_events = event_counts.gt(0)
    has_target_examples = present_events.sum(dim=1).gt(0)

    target_positions = sequence_positions.gather(1, safe_write)
    distances = target_positions - sequence_positions
    if bool((distances[valid] < 0).any()):
        raise RuntimeError("bank NMP produced a negative linguistic distance")
    distance_masks = {
        "0": valid & distances.eq(0),
        "1": valid & distances.eq(1),
        "2_4": valid & distances.ge(2) & distances.le(4),
        "5_8": valid & distances.ge(5) & distances.le(8),
        "9_16": valid & distances.ge(9) & distances.le(16),
        "17_32": valid & distances.ge(17) & distances.le(32),
        "33_plus": valid & distances.ge(33),
    }
    target_rms, target_feature_std = target_diagnostics(targets, valid)
    return BankNMPAlignment(
        targets,
        valid,
        safe_write,
        event_counts,
        present_events,
        has_target_examples,
        distance_masks,
        target_rms,
        target_feature_std,
    )


def recurrent_nmp_pass_loss(
    predictions: torch.Tensor,
    *,
    alignment: RecurrentNMPAlignment | None = None,
    final_targets: torch.Tensor | None = None,
    ordinary_mask: torch.Tensor | None = None,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth-L1 loss from each ordinary ``t`` to final memory at ordinary ``t+1``."""

    if alignment is None:
        if final_targets is None or ordinary_mask is None:
            raise ValueError("recurrent NMP requires alignment or raw target inputs")
        alignment = prepare_recurrent_nmp_alignment(
            final_targets, ordinary_mask=ordinary_mask
        )
    if predictions.shape != alignment.targets.shape:
        raise ValueError("recurrent NMP predictions and targets must have equal [B,T,D] shape")
    per_query = F.smooth_l1_loss(
        predictions.float(), alignment.targets.float(), reduction="none"
    ).mean(dim=-1)
    loss = _masked_mean(per_query, alignment.valid)
    if diagnostics is not None:
        error_rms, linear_fraction = _error_diagnostics(
            predictions, alignment.targets, alignment.valid
        )
        diagnostics.update(
            {
                "error_rms": error_rms,
                "linear_fraction": linear_fraction,
                "valid_queries": alignment.valid.sum().float(),
                "valid_events": alignment.valid.sum().float(),
            }
        )
    return loss, alignment.target_rms, alignment.target_feature_std


def bank_nmp_pass_loss(
    predictions: torch.Tensor,
    *,
    alignment: BankNMPAlignment | None = None,
    final_written_states: torch.Tensor | None = None,
    ordinary_mask: torch.Tensor | None = None,
    write_mask: torch.Tensor | None = None,
    sequence_positions: torch.Tensor | None = None,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Target-balanced loss to the first strictly-future written memory."""

    if alignment is None:
        if (
            final_written_states is None
            or ordinary_mask is None
            or write_mask is None
            or sequence_positions is None
        ):
            raise ValueError("bank NMP requires alignment or all raw alignment inputs")
        alignment = prepare_bank_nmp_alignment(
            final_written_states,
            ordinary_mask=ordinary_mask,
            write_mask=write_mask,
            sequence_positions=sequence_positions,
        )
    if predictions.shape != alignment.targets.shape:
        raise ValueError("bank NMP predictions and targets must have equal [B,T,D] shape")

    per_query = F.smooth_l1_loss(
        predictions.float(), alignment.targets.float(), reduction="none"
    ).mean(dim=-1)
    batch, length, _ = predictions.shape
    event_sums = per_query.new_zeros((batch, length)).scatter_add(
        1, alignment.safe_write, per_query * alignment.valid
    )
    event_means = event_sums / alignment.event_counts.to(per_query.dtype).clamp_min(1)
    example_counts = alignment.present_events.sum(dim=1)
    example_means = (event_means * alignment.present_events).sum(dim=1) / example_counts.clamp_min(1)
    example_loss = _masked_mean(example_means, alignment.has_target_examples)
    loss = torch.where(
        alignment.has_target_examples.any(),
        example_loss,
        predictions.float().sum() * 0.0,
    )
    distance_losses = {
        name: _masked_mean(per_query, mask)
        for name, mask in alignment.distance_masks.items()
    }
    if diagnostics is not None:
        error_rms, linear_fraction = _error_diagnostics(
            predictions, alignment.targets, alignment.valid
        )
        diagnostics.update(
            {
                "error_rms": error_rms,
                "linear_fraction": linear_fraction,
                "valid_queries": alignment.valid.sum().float(),
                "valid_events": alignment.present_events.sum().float(),
            }
        )
    return loss, alignment.target_rms, alignment.target_feature_std, distance_losses
