from __future__ import annotations

from ..variants.base import ExperimentalVariant


def configure_phase(model: ExperimentalVariant, phase: str) -> int:
    """Apply phase trainability and return the trainable parameter count."""
    return configure_trainability(
        model,
        phase=phase,
        unique_tokens_seen=0,
        freeze_pretrained_until_tokens=0,
    )


def configure_trainability(
    model: ExperimentalVariant,
    *,
    phase: str,
    unique_tokens_seen: int,
    freeze_pretrained_until_tokens: int,
) -> int:
    """Apply static phase semantics plus an integrated retrofit freeze window."""
    model.set_phase(phase)
    if (
        phase == "B"
        and int(unique_tokens_seen) < int(freeze_pretrained_until_tokens)
    ):
        added_ids = {id(parameter) for parameter in model.added_parameters()}
        for parameter in model.parameters():
            if id(parameter) not in added_ids:
                parameter.requires_grad_(False)
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
