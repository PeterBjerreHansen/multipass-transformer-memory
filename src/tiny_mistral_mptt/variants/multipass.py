from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from ..feedback import HybridPassSource
from ..nmp import (
    NMP_TARGET_NORMALIZATIONS,
    LatentPredictionHead,
    normalize_nmp_target,
    prepare_recurrent_nmp_alignment,
    prepare_bank_nmp_alignment,
    recurrent_nmp_pass_loss,
    bank_nmp_pass_loss,
)
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
    supports_recurrent_nmp = False
    supports_bank_nmp = False

    def __init__(self, backbone: MistralForCausalLM):
        super().__init__()
        self.backbone = backbone
        # Keeping disabled heads as None preserves historical state_dicts
        # exactly. configure_nmp creates only objectives with non-zero weight.
        self.recurrent_nmp_predictor: LatentPredictionHead | None = None
        self.bank_nmp_predictor: LatentPredictionHead | None = None
        self.recurrent_nmp_weight = 0.0
        self.bank_nmp_weight = 0.0
        self.recurrent_nmp_target_normalization = "rms"

    def configure_nmp(
        self,
        *,
        recurrent_weight: float,
        bank_weight: float,
        recurrent_target_normalization: str = "rms",
        projection_factor: float,
        initialization_seed: int,
    ) -> None:
        if (
            self.recurrent_nmp_predictor is not None
            or self.bank_nmp_predictor is not None
        ):
            raise RuntimeError("NMP predictors have already been configured")
        for name, value in (
            ("recurrent_weight", recurrent_weight),
            ("bank_weight", bank_weight),
        ):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if recurrent_target_normalization not in NMP_TARGET_NORMALIZATIONS:
            raise ValueError(
                "recurrent_target_normalization must be one of "
                f"{sorted(NMP_TARGET_NORMALIZATIONS)}"
            )
        if recurrent_weight and not self.supports_recurrent_nmp:
            raise ValueError(f"{self.variant_name} does not expose a recurrent NMP target")
        if bank_weight and not self.supports_bank_nmp:
            raise ValueError(f"{self.variant_name} does not expose a bank NMP target")
        hidden_size = int(self.config.hidden_size)
        kwargs = {
            "hidden_size": hidden_size,
            "projection_factor": float(projection_factor),
            "rms_norm_eps": float(self.config.rms_norm_eps),
        }
        if recurrent_weight:
            self.recurrent_nmp_predictor = LatentPredictionHead(
                **kwargs, initialization_seed=int(initialization_seed) + 100_003
            )
        if bank_weight:
            self.bank_nmp_predictor = LatentPredictionHead(
                **kwargs, initialization_seed=int(initialization_seed) + 200_003
            )
        self.recurrent_nmp_weight = float(recurrent_weight)
        self.bank_nmp_weight = float(bank_weight)
        self.recurrent_nmp_target_normalization = str(recurrent_target_normalization)

    def added_parameters(self):
        if self.recurrent_nmp_predictor is not None:
            yield from self.recurrent_nmp_predictor.parameters()
        if self.bank_nmp_predictor is not None:
            yield from self.bank_nmp_predictor.parameters()

    def initialization_only_state_prefixes(self) -> tuple[str, ...]:
        """State prefixes allowed to be absent from an ``init_from`` checkpoint."""
        prefixes: list[str] = []
        if self.recurrent_nmp_predictor is not None:
            prefixes.append("recurrent_nmp_predictor.")
        if self.bank_nmp_predictor is not None:
            prefixes.append("bank_nmp_predictor.")
        return tuple(prefixes)

    @staticmethod
    def _source_component(run: HiddenRun, component: str) -> torch.Tensor:
        source = run.feedback_source
        if isinstance(source, HybridPassSource):
            return (
                source.recurrent_hidden
                if component == "recurrent"
                else source.bank_hidden
            )
        if component == "recurrent":
            return source
        return run.hidden_states

    def nmp_written_states(self, final_bank_source: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(f"{self.variant_name} does not implement bank NMP writes")

    def nmp_write_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(f"{self.variant_name} does not implement a bank write policy")

    def nmp_sequence_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(f"{self.variant_name} does not implement bank positions")

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
        recurrent_nmp_loss_weights: Sequence[float] | None = None,
        bank_nmp_loss_weights: Sequence[float] | None = None,
        nmp_weight_scale: float = 1.0,
    ) -> TrainOutput:
        if not 0.0 <= float(nmp_weight_scale) <= 1.0:
            raise ValueError("nmp_weight_scale must lie in [0,1]")
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
        auxiliary_loss = loss.new_zeros(())
        ordinary_mask = ~self.control_token_mask(input_ids)

        if self.recurrent_nmp_predictor is not None:
            with torch.no_grad():
                final_target = normalize_nmp_target(
                    self._source_component(runs[-1], "recurrent"),
                    normalization=self.recurrent_nmp_target_normalization,
                    eps=float(self.config.rms_norm_eps),
                ).detach()
                recurrent_alignment = prepare_recurrent_nmp_alignment(
                    final_target,
                    ordinary_mask=ordinary_mask,
                )
            recurrent_pass_weights = normalize_pass_weights(
                recurrent_nmp_loss_weights,
                passes,
                device=loss.device,
                dtype=loss.dtype,
            )
            nmp_losses: list[torch.Tensor] = []
            nmp_diagnostics: list[dict[str, torch.Tensor]] = []
            for index, run in enumerate(runs):
                # The predictor sees only h_t. No token embedding or x_{t+1}
                # exists anywhere on this branch.
                prediction = self.recurrent_nmp_predictor(run.hidden_states)
                pass_diagnostics: dict[str, torch.Tensor] = {}
                nmp_loss, _, _ = recurrent_nmp_pass_loss(
                    prediction,
                    alignment=recurrent_alignment,
                    diagnostics=pass_diagnostics,
                )
                nmp_losses.append(nmp_loss)
                nmp_diagnostics.append(pass_diagnostics)
            raw = sum(
                weight * pass_loss
                for weight, pass_loss in zip(
                    recurrent_pass_weights, nmp_losses, strict=True
                )
            )
            weighted = raw * self.recurrent_nmp_weight * float(nmp_weight_scale)
            auxiliary_loss = auxiliary_loss + weighted
            pass_loss_values = torch.stack(nmp_losses).detach().cpu().tolist()
            pass_weight_values = recurrent_pass_weights.detach().cpu().tolist()
            # Transfer each diagnostic vector once per objective, rather than
            # synchronizing separately for every pass.
            error_rms_values = torch.stack(
                [item["error_rms"] for item in nmp_diagnostics]
            ).detach().cpu().tolist()
            linear_fraction_values = torch.stack(
                [item["linear_fraction"] for item in nmp_diagnostics]
            ).detach().cpu().tolist()
            for index, (pass_loss, pass_weight, error_rms, linear_fraction) in enumerate(
                zip(
                    pass_loss_values,
                    pass_weight_values,
                    error_rms_values,
                    linear_fraction_values,
                    strict=True,
                )
            ):
                metrics[f"recurrent_nmp_pass_{index + 1}_loss"] = pass_loss
                metrics[f"recurrent_nmp_pass_{index + 1}_weight"] = pass_weight
                metrics[f"recurrent_nmp_pass_{index + 1}_error_rms"] = error_rms
                metrics[f"recurrent_nmp_pass_{index + 1}_linear_fraction"] = linear_fraction
            recurrent_valid_count = float(
                recurrent_alignment.valid.sum().detach().cpu()
            )
            metrics.update(
                {
                    "recurrent_nmp_loss": float(raw.detach().cpu()),
                    "recurrent_nmp_weighted_loss": float(weighted.detach().cpu()),
                    "recurrent_nmp_target_rms": float(
                        recurrent_alignment.target_rms.detach().cpu()
                    ),
                    "recurrent_nmp_target_feature_std": float(
                        recurrent_alignment.target_feature_std.detach().cpu()
                    ),
                    "recurrent_nmp_valid_queries": recurrent_valid_count,
                    "recurrent_nmp_valid_events": recurrent_valid_count,
                }
            )

        if self.bank_nmp_predictor is not None:
            final_bank_source = self._source_component(runs[-1], "bank")
            # Stop-gradient covers both the final source and the writer target
            # branch. The same writer can still receive gradients through any
            # memories that causally contributed to run.hidden_states.
            with torch.no_grad():
                written_targets = self.nmp_written_states(final_bank_source).detach()
            write_mask = self.nmp_write_mask(input_ids)
            sequence_positions = self.nmp_sequence_positions(input_ids)
            with torch.no_grad():
                bank_alignment = prepare_bank_nmp_alignment(
                    written_targets,
                    ordinary_mask=ordinary_mask,
                    write_mask=write_mask,
                    sequence_positions=sequence_positions,
                )
            bank_pass_weights = normalize_pass_weights(
                bank_nmp_loss_weights,
                passes,
                device=loss.device,
                dtype=loss.dtype,
            )
            nmp_losses = []
            nmp_diagnostics = []
            distance_values: dict[str, list[torch.Tensor]] = {
                name: [] for name in bank_alignment.distance_masks
            }
            for index, run in enumerate(runs):
                prediction = self.bank_nmp_predictor(run.hidden_states)
                pass_diagnostics = {}
                nmp_loss, _, _, distances = bank_nmp_pass_loss(
                    prediction,
                    alignment=bank_alignment,
                    diagnostics=pass_diagnostics,
                )
                nmp_losses.append(nmp_loss)
                for name, value in distances.items():
                    distance_values[name].append(value)
                nmp_diagnostics.append(pass_diagnostics)
            raw = sum(
                weight * pass_loss
                for weight, pass_loss in zip(
                    bank_pass_weights, nmp_losses, strict=True
                )
            )
            weighted = raw * self.bank_nmp_weight * float(nmp_weight_scale)
            auxiliary_loss = auxiliary_loss + weighted
            pass_loss_values = torch.stack(nmp_losses).detach().cpu().tolist()
            pass_weight_values = bank_pass_weights.detach().cpu().tolist()
            error_rms_values = torch.stack(
                [item["error_rms"] for item in nmp_diagnostics]
            ).detach().cpu().tolist()
            linear_fraction_values = torch.stack(
                [item["linear_fraction"] for item in nmp_diagnostics]
            ).detach().cpu().tolist()
            for index, (pass_loss, pass_weight, error_rms, linear_fraction) in enumerate(
                zip(
                    pass_loss_values,
                    pass_weight_values,
                    error_rms_values,
                    linear_fraction_values,
                    strict=True,
                )
            ):
                metrics[f"bank_nmp_pass_{index + 1}_loss"] = pass_loss
                metrics[f"bank_nmp_pass_{index + 1}_weight"] = pass_weight
                metrics[f"bank_nmp_pass_{index + 1}_error_rms"] = error_rms
                metrics[f"bank_nmp_pass_{index + 1}_linear_fraction"] = linear_fraction
            metrics.update(
                {
                    "bank_nmp_loss": float(raw.detach().cpu()),
                    "bank_nmp_weighted_loss": float(weighted.detach().cpu()),
                    "bank_nmp_target_rms": float(
                        bank_alignment.target_rms.detach().cpu()
                    ),
                    "bank_nmp_target_feature_std": float(
                        bank_alignment.target_feature_std.detach().cpu()
                    ),
                    "bank_nmp_valid_queries": float(
                        bank_alignment.valid.sum().detach().cpu()
                    ),
                    "bank_nmp_valid_events": float(
                        bank_alignment.present_events.sum().detach().cpu()
                    ),
                }
            )
            distance_names = list(distance_values)
            distance_matrix = torch.stack(
                [torch.stack(distance_values[name]) for name in distance_names]
            )
            distance_means = distance_matrix.detach().cpu().mean(dim=1).tolist()
            metrics.update(
                {
                    f"bank_nmp_distance_{name}_loss": value
                    for name, value in zip(distance_names, distance_means, strict=True)
                }
            )

        if (
            self.recurrent_nmp_predictor is not None
            or self.bank_nmp_predictor is not None
        ):
            metrics["nmp_weight_scale"] = float(nmp_weight_scale)
            metrics["ntp_loss"] = float(loss.detach().cpu())
        loss = loss + auxiliary_loss
        return TrainOutput(
            loss=loss,
            pass_losses=tuple(pass_losses),
            effective_passes=passes,
            metrics=metrics,
        )

    # Default public model semantics remain the exact one-pass vanilla model.
    # Multipass evaluation is explicit through compute_passes/pass-depth tools.
    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.backbone.generate(*args, **kwargs)
