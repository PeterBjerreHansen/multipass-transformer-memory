from __future__ import annotations

from typing import Literal

import torch

from ..variants.multipass import MultiPassVariant
from .state import ExactIncrementalState, PassStreamState, RecurrentState

InferenceMode = Literal["exact_incremental", "recurrent"]


def _validate_prompt(input_ids: torch.Tensor, passes: int) -> None:
    if input_ids.ndim != 2 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must be non-empty [B,T]")
    if passes < 1:
        raise ValueError("passes must be positive")


def _validate_token(token: torch.Tensor, batch_size: int) -> None:
    if token.ndim != 2 or token.shape != (batch_size, 1):
        raise ValueError(f"token must have shape [{batch_size},1]")


@torch.no_grad()
def prefill_exact(
    model: MultiPassVariant,
    input_ids: torch.Tensor,
    *,
    passes: int,
) -> ExactIncrementalState:
    """Build an exact cached K-pass state for an arbitrary positive ``K``.

    Every pass owns its own TinyMistral KV stream.  The feedback memory stored
    for stream ``k`` is the strict-past source that stream ``k+1`` will use on
    the next appended token.  This state therefore reproduces full-prefix
    K-pass recomputation while requiring only K one-token backbone steps per
    continuation token.
    """
    _validate_prompt(input_ids, passes)
    if passes > 1 and not model.supports_cached_feedback:
        raise ValueError(
            f"{model.variant_name} does not implement cached feedback inference"
        )

    first_run = model._run_first_state_cached(input_ids)
    if first_run.past_key_values is None:
        raise RuntimeError("cached first pass did not return KV state")
    runs = [first_run]

    if passes > 1:
        token_embeddings = model.input_embeddings(input_ids)
        previous = first_run.feedback_source
        for _ in range(1, passes):
            run = model._run_feedback_state_cached(
                input_ids,
                token_embeddings,
                previous,
            )
            if run.past_key_values is None:
                raise RuntimeError("cached feedback pass did not return KV state")
            runs.append(run)
            previous = run.feedback_source

    if passes == 1:
        # K=1 is the SWA Transformer cached boundary and must not require a variant to
        # implement any feedback-state protocol. The placeholder memory is not
        # consumed while feedback is disabled.
        streams = (
            PassStreamState(
                past_key_values=first_run.past_key_values,
                feedback_memory=first_run.hidden_states[:, -1:, :].detach(),
                last_hidden=first_run.hidden_states[:, -1:, :].detach(),
            ),
        )
    else:
        streams = tuple(
            PassStreamState(
                past_key_values=run.past_key_values,
                feedback_memory=model._feedback_memory_from_hidden(
                    run.feedback_source, input_ids=input_ids
                ),
                last_hidden=run.hidden_states[:, -1:, :].detach(),
            )
            for run in runs
        )
    prediction_hidden = model.prediction_hidden_after_sequence(
        runs[-1].hidden_states, input_ids
    )
    logits = model.backbone.lm_head(prediction_hidden).float()[:, -1, :]
    return ExactIncrementalState(
        prefill_passes=passes,
        streams=streams,
        next_token_logits=logits,
    )


def recurrent_from_exact(state: ExactIncrementalState) -> RecurrentState:
    """Collapse an already-prefilled exact K-pass state without recomputation."""
    final_stream = state.streams[-1]
    feedback_memory = (
        None
        if state.prefill_passes == 1
        else state.streams[-2].feedback_memory
    )
    return RecurrentState(
        prefill_passes=state.prefill_passes,
        past_key_values=final_stream.past_key_values,
        feedback_memory=feedback_memory,
        last_hidden=final_stream.last_hidden,
        next_token_logits=state.next_token_logits,
    )


@torch.no_grad()
def prefill_recurrent(
    model: MultiPassVariant,
    input_ids: torch.Tensor,
    *,
    passes: int,
) -> RecurrentState:
    """Prefill K passes, then collapse to one continuing recurrent stream.

    For K>1 the final-pass KV cache is paired with feedback memory from pass
    K-1.  Consequently the first appended-token processing step is exactly the
    same as exact K-pass inference.  Only after that step does the model close
    the loop by feeding its newly produced final-pass state back to itself.

    K=1 is a deliberate boundary case: feedback remains disabled and the state
    is ordinary SWA Transformer cached TinyMistral inference.
    """
    return recurrent_from_exact(prefill_exact(model, input_ids, passes=passes))


@torch.no_grad()
def exact_decode_step(
    model: MultiPassVariant,
    state: ExactIncrementalState,
    token: torch.Tensor,
) -> ExactIncrementalState:
    """Consume one observed token while preserving exact K-pass semantics."""
    batch_size = state.next_token_logits.shape[0]
    _validate_token(token, batch_size)

    # Crucial causality rule: all pass-k computations below read the *old*
    # stream-(k-1) memory.  Newly produced same-position states are not exposed
    # until every stream has completed this token.
    position = state.next_position
    token_embedding = model.input_embeddings(token)

    first_run = model._run_first_token_state_cached(
        token,
        state.streams[0].past_key_values,
    )
    new_runs = [first_run]

    for pass_index in range(1, state.prefill_passes):
        run = model._run_feedback_token_state_cached(
            token_embedding,
            state.streams[pass_index - 1].feedback_memory,
            state.streams[pass_index].past_key_values,
            token=token,
        )
        new_runs.append(run)

    if state.prefill_passes == 1:
        streams = (
            PassStreamState(
                past_key_values=first_run.past_key_values,
                feedback_memory=first_run.hidden_states[:, -1:, :].detach(),
                last_hidden=first_run.hidden_states[:, -1:, :].detach(),
            ),
        )
    else:
        streams = tuple(
            PassStreamState(
                past_key_values=run.past_key_values,
                feedback_memory=model._append_feedback_memory(
                    old_stream.feedback_memory,
                    run.feedback_source,
                    token=token,
                    position=position,
                ),
                last_hidden=run.hidden_states[:, -1:, :].detach(),
            )
            for old_stream, run in zip(
                state.streams,
                new_runs,
                strict=True,
            )
        )
    candidate_logits = model.backbone.lm_head(
        new_runs[-1].hidden_states[:, -1:, :]
    ).float()[:, -1, :]
    control = model.control_token_mask(token)[:, 0]
    logits = torch.where(control[:, None], state.next_token_logits, candidate_logits)
    return ExactIncrementalState(
        prefill_passes=state.prefill_passes,
        streams=streams,
        next_token_logits=logits,
    )


@torch.no_grad()
def recurrent_decode_step(
    model: MultiPassVariant,
    state: RecurrentState,
    token: torch.Tensor,
) -> RecurrentState:
    """Consume one observed token in the collapsed one-stream recurrence."""
    batch_size = state.next_token_logits.shape[0]
    _validate_token(token, batch_size)

    position = state.next_position
    if not state.feedback_enabled:
        run = model._run_first_token_state_cached(token, state.past_key_values)
        feedback_memory = None
    else:
        assert state.feedback_memory is not None
        token_embedding = model.input_embeddings(token)
        run = model._run_feedback_token_state_cached(
            token_embedding,
            state.feedback_memory,
            state.past_key_values,
            token=token,
        )
        feedback_memory = model._append_feedback_memory(
            state.feedback_memory,
            run.feedback_source,
            token=token,
            position=position,
        )

    candidate_logits = model.backbone.lm_head(
        run.hidden_states[:, -1:, :]
    ).float()[:, -1, :]
    control = model.control_token_mask(token)[:, 0]
    logits = torch.where(control[:, None], state.next_token_logits, candidate_logits)
    return RecurrentState(
        prefill_passes=state.prefill_passes,
        past_key_values=run.past_key_values,
        feedback_memory=feedback_memory,
        last_hidden=run.hidden_states[:, -1:, :].detach(),
        next_token_logits=logits,
    )


@torch.no_grad()
def prefill(
    model: MultiPassVariant,
    input_ids: torch.Tensor,
    *,
    passes: int,
    mode: InferenceMode,
) -> ExactIncrementalState | RecurrentState:
    if mode == "exact_incremental":
        return prefill_exact(model, input_ids, passes=passes)
    if mode == "recurrent":
        return prefill_recurrent(model, input_ids, passes=passes)
    raise ValueError(f"unknown inference mode {mode!r}")


@torch.no_grad()
def decode_step(
    model: MultiPassVariant,
    state: ExactIncrementalState | RecurrentState,
    token: torch.Tensor,
) -> ExactIncrementalState | RecurrentState:
    if isinstance(state, ExactIncrementalState):
        return exact_decode_step(model, state, token)
    if isinstance(state, RecurrentState):
        return recurrent_decode_step(model, state, token)
    raise TypeError(f"unsupported inference state {type(state)!r}")
