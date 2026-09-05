from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from ..feedback import HybridPassSource
from ..training.loss import causal_lm_loss_from_labels, normalize_pass_weights
from .base import ExperimentalVariant, TrainOutput


@dataclass(frozen=True)
class HiddenRun:
    """Final hidden states plus the architecture-specific recurrence source."""

    hidden_states: torch.Tensor
    feedback_source: torch.Tensor | HybridPassSource
    past_key_values: tuple[LayerKVCache, ...] | None = None


@dataclass(frozen=True)
class PassResult:
    hidden_states: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class MultiPassResult:
    passes: tuple[PassResult, ...]

    @property
    def final(self) -> PassResult:
        return self.passes[-1]


def shift_previous_hidden(previous_hidden: torch.Tensor) -> torch.Tensor:
    """Right-shift a [B,T,D] previous-pass state by exactly one token.

    Position zero has no causal predecessor and is therefore filled with zeros.
    This helper defines the shared alignment contract for single-state feedback
    variants such as FBT and MemoryAdd.
    """
    if previous_hidden.ndim != 3:
        raise ValueError("previous_hidden must be [B,T,D]")
    shifted = torch.zeros_like(previous_hidden)
    if previous_hidden.shape[1] > 1:
        shifted[:, 1:, :] = previous_hidden[:, :-1, :]
    return shifted


class MultiPassVariant(ExperimentalVariant):
    """Shared pass recurrence and objective plumbing for research variants.

    Architectures define how pass ``k>1`` consumes the previous pass's final
    top-layer states. Ordinary-token variants use the validated vanilla
    TinyMistral input path on pass 1. Architectures with input-only control
    positions may additionally supply control embeddings and self-attention key
    masks while retaining the same pretrained backbone weights.
    """
    supports_cached_feedback = False

    def __init__(self, backbone: MistralForCausalLM):
        super().__init__()
        self.backbone = backbone

    @property
    def config(self):
        return self.backbone.config

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed model-input IDs, including architecture control IDs when present."""
        return self.backbone.model.embed_tokens(input_ids)

    def self_attention_key_mask(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        """Optional bool [B,T] mask controlling which positions persist as self-attention K/V."""
        return None

    def build_lm_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return position-aligned LM labels; -100 marks non-prediction positions."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        labels = torch.full_like(input_ids, -100)
        if input_ids.shape[1] > 1:
            labels[:, :-1] = input_ids[:, 1:]
        return labels

    def lm_loss(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return causal_lm_loss_from_labels(logits, self.build_lm_labels(input_ids))

    def control_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Bool [B,T] positions that are architectural controls, not language."""
        return torch.zeros_like(input_ids, dtype=torch.bool)

    def prediction_hidden_after_sequence(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Hidden state whose logits predict the next linguistic token."""
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("hidden_states/input_ids token shapes differ")
        return hidden_states[:, -1:, :]

    def phase_a_first_pass_requires_grad(self) -> bool:
        """Whether an added parameter participates inside pass 1 in Phase A."""
        return False

    def _run_first_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.model(
            inputs_embeds=self.input_embeddings(input_ids),
            attention_mask=self.self_attention_key_mask(input_ids),
            use_cache=False,
        ).last_hidden_state

    def _run_first_hidden_cached(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        output = self.backbone.model(
            inputs_embeds=self.input_embeddings(input_ids),
            attention_mask=self.self_attention_key_mask(input_ids),
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("cached first pass did not return KV state")
        return output.last_hidden_state, output.past_key_values

    def _run_first_token_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        output = self.backbone.model(
            inputs_embeds=self.input_embeddings(input_ids),
            attention_mask=self.self_attention_key_mask(input_ids),
            past_key_values=past_key_values,
            use_cache=True,
        )
        if output.past_key_values is None:
            raise RuntimeError("cached first-pass token did not return KV state")
        return output.last_hidden_state, output.past_key_values

    # Rich state hooks preserve the legacy hidden-only API while allowing a
    # variant to feed back an internal state, such as a source decoder layer,
    # instead of its final normalized hidden state.
    def _run_first_state(self, input_ids: torch.Tensor) -> HiddenRun:
        hidden = self._run_first_hidden(input_ids)
        return HiddenRun(hidden, hidden)

    def _run_feedback_state(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_source: torch.Tensor,
    ) -> HiddenRun:
        hidden = self._run_feedback_hidden(input_ids, token_embeddings, previous_source)
        return HiddenRun(hidden, hidden)

    def run_feedback_transition(
        self,
        input_ids: torch.Tensor,
        previous_source: torch.Tensor | HybridPassSource,
        *,
        bypass: bool = False,
    ) -> HiddenRun:
        """Run one explicit pass transition through the variant boundary.

        Intervention tools use this seam so they do not need to know how a
        particular reader/merger is wired. ``bypass=True`` executes the
        architecture's ordinary first-pass path, including any control-token
        masking, while omitting the feedback pathway completely.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be non-empty [B,T]")
        if bypass:
            return self._run_first_state(input_ids)
        return self._run_feedback_state(
            input_ids,
            self.input_embeddings(input_ids),
            previous_source,
        )

    def _run_first_state_cached(self, input_ids: torch.Tensor) -> HiddenRun:
        hidden, cache = self._run_first_hidden_cached(input_ids)
        return HiddenRun(hidden, hidden, cache)

    def _run_feedback_state_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_source: torch.Tensor,
    ) -> HiddenRun:
        hidden, cache = self._run_feedback_hidden_cached(
            input_ids, token_embeddings, previous_source
        )
        return HiddenRun(hidden, hidden, cache)

    def _run_first_token_state_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> HiddenRun:
        hidden, cache = self._run_first_token_cached(input_ids, past_key_values)
        return HiddenRun(hidden, hidden, cache)

    def _run_feedback_token_state_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> HiddenRun:
        hidden, cache = self._run_feedback_token_cached(
            token_embedding,
            feedback_memory,
            past_key_values,
            token=token,
        )
        return HiddenRun(hidden, hidden, cache)

    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        """Run a complete feedback pass while retaining its self-attention cache."""
        raise NotImplementedError

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        """Process one token from an already-strict-past feedback memory."""
        raise NotImplementedError

    def _feedback_memory_from_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ):
        """Compress a stream history to the state needed by the next pass/token."""
        raise NotImplementedError

    def _append_feedback_memory(
        self,
        feedback_memory,
        new_hidden: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ):
        """Append one newly produced stream state to its retained feedback memory."""
        raise NotImplementedError

    def _run_passes(
        self,
        input_ids: torch.Tensor,
        *,
        passes: int,
        phase: str,
    ) -> tuple[HiddenRun, ...]:
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("input_ids must be [B,T] with at least two tokens")
        if passes < 1:
            raise ValueError("passes must be positive")
        if phase not in {"A", "B"}:
            raise ValueError("phase must be 'A' or 'B'")
        if phase == "A" and passes < 2:
            raise ValueError("Phase A requires at least two passes")

        # Most Phase-A variants have no added parameter inside pass 1, so its
        # frozen-backbone graph can be discarded. Memory-token variants are the
        # deliberate exception: their learned input-only <MEM> embedding is an
        # added parameter and must receive gradients through the pass-1 state
        # that is later written/read by the recurrent pathway.
        if phase == "A" and not self.phase_a_first_pass_requires_grad():
            with torch.no_grad():
                first_run = self._run_first_state(input_ids)
        else:
            first_run = self._run_first_state(input_ids)

        runs = [first_run]
        if passes == 1:
            return tuple(runs)

        token_embeddings = self.input_embeddings(input_ids)
        previous = first_run.feedback_source
        for _ in range(1, passes):
            run = self._run_feedback_state(input_ids, token_embeddings, previous)
            runs.append(run)
            previous = run.feedback_source
        return tuple(runs)

    def _run_hidden_passes(
        self,
        input_ids: torch.Tensor,
        *,
        passes: int,
        phase: str,
    ) -> tuple[torch.Tensor, ...]:
        """Compatibility view retained for callers that need only top states."""
        return tuple(
            run.hidden_states
            for run in self._run_passes(input_ids, passes=passes, phase=phase)
        )

    def compute_passes(
        self,
        input_ids: torch.Tensor,
        *,
        passes: int,
        phase: str = "B",
    ) -> MultiPassResult:
        hidden_states = self._run_hidden_passes(input_ids, passes=passes, phase=phase)
        return MultiPassResult(
            tuple(
                PassResult(
                    hidden_states=hidden, logits=self.backbone.lm_head(hidden).float()
                )
                for hidden in hidden_states
            )
        )

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        *,
        phase: str = "B",
        passes: int = 1,
        loss_weights: Sequence[float] | None = None,
    ) -> TrainOutput:
        runs = self._run_passes(input_ids, passes=passes, phase=phase)
        hidden_states = tuple(run.hidden_states for run in runs)
        pass_losses: list[torch.Tensor] = []
        for hidden in hidden_states:
            logits = self.backbone.lm_head(hidden).float()
            pass_losses.append(self.lm_loss(logits, input_ids))

        weights = normalize_pass_weights(
            loss_weights,
            passes,
            device=pass_losses[-1].device,
            dtype=pass_losses[-1].dtype,
        )
        loss = sum(
            weight * pass_loss
            for weight, pass_loss in zip(weights, pass_losses, strict=True)
        )
        metrics = {
            f"pass_{index + 1}_loss": float(pass_loss.detach().cpu())
            for index, pass_loss in enumerate(pass_losses)
        }
        metrics.update(
            {
                f"pass_{index + 1}_weight": float(weight.detach().cpu())
                for index, weight in enumerate(weights)
            }
        )
        return TrainOutput(
            loss=loss,
            pass_losses=tuple(pass_losses),
            effective_passes=passes,
            metrics=metrics,
        )

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "MultiPassVariant.forward() has no unambiguous temporal semantics. "
            "Use compute_passes(...), prefill_exact_k_pass(...), or "
            "prefill_live_feedback(...) explicitly."
        )

    def generate(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "MultiPassVariant.generate() has no unambiguous temporal semantics. "
            "Use the explicit exact-K-pass or live-feedback inference helpers."
        )
