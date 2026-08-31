from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from tiny_mistral.modeling import LayerKVCache, MistralForCausalLM

from .base import TrainOutput
from .multipass import HiddenRun, MultiPassVariant, shift_previous_hidden


@dataclass(frozen=True)
class _RecirculationRun:
    """One token/pass run plus the residuals needed by Eq. 1."""

    hidden_states: torch.Tensor
    source_state: torch.Tensor
    destination_state: torch.Tensor
    past_key_values: tuple[LayerKVCache, ...] | None

    def hidden_run(self) -> HiddenRun:
        return HiddenRun(
            hidden_states=self.hidden_states,
            feedback_source=self.source_state,
            past_key_values=self.past_key_values,
        )


class _AdaptiveRecirculationController(nn.Module):
    """Predict token- and feature-wise recirculation coefficients.

    This follows Mozer et al.'s conditional vector alpha/beta controller: a
    two-hidden-layer GELU MLP consumes the concatenated source and destination
    residual streams and predicts independent sigmoid-bounded coefficient
    vectors. The zero output-weight initialization makes the controller start
    at the fixed convex mixture and lets the coefficient head learn first.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        initial_alpha: float,
        initialization_seed: int,
    ):
        super().__init__()
        input_size = 2 * int(hidden_size)
        hidden_size = int(hidden_size)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.input_norm = nn.LayerNorm(input_size)
            self.hidden_1 = nn.Linear(input_size, hidden_size)
            self.hidden_2 = nn.Linear(hidden_size, hidden_size)
            self.output = nn.Linear(hidden_size, 2 * hidden_size)
            nn.init.xavier_uniform_(self.hidden_1.weight)
            nn.init.zeros_(self.hidden_1.bias)
            nn.init.xavier_uniform_(self.hidden_2.weight)
            nn.init.zeros_(self.hidden_2.bias)
            # At initialization, the adaptive controller is the existing
            # fixed-alpha recirculation rule. Sigmoid cannot represent exact
            # endpoints, so clamp only for the initialization logit.
            safe_alpha = min(max(float(initial_alpha), 1e-6), 1.0 - 1e-6)
            safe_beta = min(max(1.0 - float(initial_alpha), 1e-6), 1.0 - 1e-6)
            alpha_logit = math.log(safe_alpha / (1.0 - safe_alpha))
            beta_logit = math.log(safe_beta / (1.0 - safe_beta))
            nn.init.zeros_(self.output.weight)
            nn.init.constant_(self.output.bias[:hidden_size], alpha_logit)
            nn.init.constant_(self.output.bias[hidden_size:], beta_logit)

    def forward(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((source, destination), dim=-1)
        hidden = F.gelu(self.hidden_1(self.input_norm(features)))
        hidden = F.gelu(self.hidden_2(hidden))
        logits = self.output(hidden)
        alpha_logits, beta_logits = logits.chunk(2, dim=-1)
        return torch.sigmoid(alpha_logits), torch.sigmoid(beta_logits)


class RecirculationVariant(MultiPassVariant):
    """Source-layer recirculation with fixed or adaptive mixing coefficients.

    The output of ``source_layer`` is norm-matched and mixed into the output
    of the earlier ``destination_layer`` on later passes. The source is
    right-shifted before it is consumed, so position zero has no predecessor
    and cannot receive feedback from the current sequence.

    ``mode="fixed"`` preserves the original scalar convex mixture. ``mode="adaptive"``
    follows Mozer et al.'s conditional vector alpha/beta formulation: a small
    controller predicts independent per-token, per-feature coefficients from
    the source and destination states. The controller is initialized to the
    fixed mixture and is exposed as architecture-added parameters, so Phase A
    trains only the controller while Phase B can also fine-tune the backbone.
    """

    variant_name = "recirculation"
    supports_cached_feedback = True

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        source_layer: int,
        destination_layer: int,
        alpha: float = 0.1,
        mode: str = "fixed",
        initialization_seed: int = 4242,
    ):
        super().__init__(backbone)
        layer_count = len(backbone.model.layers)
        if not (0 <= destination_layer < source_layer < layer_count):
            raise ValueError(
                "require 0 <= destination_layer < source_layer < num_layers"
            )
        if not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be finite in [0,1]")
        if mode not in {"fixed", "adaptive"}:
            raise ValueError("recirculation mode must be 'fixed' or 'adaptive'")
        self.source_layer = int(source_layer)
        self.destination_layer = int(destination_layer)
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.adaptive_controller: _AdaptiveRecirculationController | None = None
        if self.mode == "adaptive":
            self.adaptive_controller = _AdaptiveRecirculationController(
                int(backbone.config.hidden_size),
                initial_alpha=self.alpha,
                initialization_seed=initialization_seed,
            )

    def added_parameters(self) -> Iterable[nn.Parameter]:
        yield from super().added_parameters()
        if self.adaptive_controller is not None:
            yield from self.adaptive_controller.parameters()

    @staticmethod
    def _norm_match(source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        source_norm = torch.linalg.vector_norm(
            source.float(), ord=2, dim=-1, keepdim=True
        )
        destination_norm = torch.linalg.vector_norm(
            destination.float(), ord=2, dim=-1, keepdim=True
        )
        scale = (destination_norm / source_norm.clamp_min(1e-12)).to(source.dtype)
        return source * scale

    def _mix(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        *,
        valid_feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source.shape != destination.shape:
            raise ValueError("recirculation source and destination shapes differ")
        matched = self._norm_match(source, destination)
        if self.adaptive_controller is None:
            candidate = self.alpha * matched + (1.0 - self.alpha) * destination
        else:
            alpha, beta = self.adaptive_controller(source, destination)
            candidate = alpha * matched + beta * destination
        if valid_feedback is None:
            return candidate
        return torch.where(valid_feedback[..., None], candidate, destination)

    @staticmethod
    def _cache_next_position(
        past_key_values: tuple[LayerKVCache, ...] | None,
    ) -> int:
        if not past_key_values:
            return 0
        positions = {cache.next_position for cache in past_key_values}
        if len(positions) != 1:
            raise ValueError("layer caches disagree on next position")
        return next(iter(positions))

    def _run_core(
        self,
        embeddings: torch.Tensor,
        *,
        recurrent_source: torch.Tensor | None,
        destination_override: torch.Tensor | None = None,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        detach_cache: bool = True,
        full_sequence_feedback: bool = False,
    ) -> _RecirculationRun:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must be [B,T,D]")
        if past_key_values is not None and len(past_key_values) != len(self.backbone.model.layers):
            raise ValueError("past_key_values must contain one cache per decoder layer")
        if recurrent_source is not None and recurrent_source.shape != embeddings.shape:
            raise ValueError("recirculation source and embeddings shapes differ")
        if destination_override is not None and destination_override.shape != embeddings.shape:
            raise ValueError("recirculation destination override and embeddings shapes differ")
        if recurrent_source is not None and destination_override is not None:
            raise ValueError("provide recurrent_source or destination_override, not both")

        batch_size, seq_len, _ = embeddings.shape
        start = self._cache_next_position(past_key_values)
        position_ids = torch.arange(
            start,
            start + seq_len,
            device=embeddings.device,
            dtype=torch.long,
        )[None, :].expand(batch_size, -1)
        valid_feedback = None
        if recurrent_source is not None and full_sequence_feedback:
            valid_feedback = torch.ones(
                (batch_size, seq_len), dtype=torch.bool, device=embeddings.device
            )
            valid_feedback[:, 0] = False

        hidden_states = embeddings
        caches: list[LayerKVCache] | None = [] if use_cache else None
        source_capture: torch.Tensor | None = None
        destination_capture: torch.Tensor | None = None
        for layer_index, layer in enumerate(self.backbone.model.layers):
            past = None if past_key_values is None else past_key_values[layer_index]
            hidden_states, cache = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past,
                use_cache=use_cache,
                detach_cache=detach_cache,
                fast_attention_compatible=past_key_values is None,
            )
            if caches is not None:
                if cache is None:
                    raise RuntimeError("recirculation layer did not return KV state")
                caches.append(cache)
            if layer_index == self.destination_layer:
                destination_capture = hidden_states
                if recurrent_source is not None:
                    hidden_states = self._mix(
                        recurrent_source,
                        hidden_states,
                        valid_feedback=valid_feedback,
                    )
                elif destination_override is not None:
                    hidden_states = destination_override
            if layer_index == self.source_layer:
                source_capture = hidden_states

        if source_capture is None or destination_capture is None:
            raise RuntimeError("recirculation source/destination layer was not reached")
        hidden_states = self.backbone.model.norm(hidden_states)
        return _RecirculationRun(
            hidden_states=hidden_states,
            source_state=source_capture,
            destination_state=destination_capture,
            past_key_values=tuple(caches) if caches is not None else None,
        )

    def _core(
        self,
        embeddings: torch.Tensor,
        *,
        recurrent_source: torch.Tensor | None,
        past_key_values: tuple[LayerKVCache, ...] | None,
        use_cache: bool,
        full_sequence_feedback: bool = False,
    ) -> HiddenRun:
        """Compatibility wrapper for the parallel-multipass implementation."""
        return self._run_core(
            embeddings,
            recurrent_source=recurrent_source,
            past_key_values=past_key_values,
            use_cache=use_cache,
            full_sequence_feedback=full_sequence_feedback,
        ).hidden_run()

    def _replay_upper_stack(
        self,
        mixed_destination: torch.Tensor,
        *,
        ordinary_caches: tuple[LayerKVCache, ...],
        previous_caches: tuple[LayerKVCache, ...] | None,
        detach_cache: bool,
    ) -> tuple[LayerKVCache, ...]:
        """Replay the current token above the destination and replace upper K/V.

        Layers through ``destination_layer`` are identical to the first
        iteration and keep its cache entries. Layers above the destination are
        recomputed from the Eq. 1 mixture against the strict-past cache.
        """
        if len(ordinary_caches) != len(self.backbone.model.layers):
            raise ValueError("ordinary_caches must contain one cache per decoder layer")
        if previous_caches is not None and len(previous_caches) != len(ordinary_caches):
            raise ValueError("previous_caches must contain one cache per decoder layer")

        batch_size, seq_len, _ = mixed_destination.shape
        if seq_len != 1:
            raise ValueError("paper recirculation replay requires one token")
        position = ordinary_caches[0].next_position - 1
        position_ids = torch.full(
            (batch_size, 1), position, dtype=torch.long, device=mixed_destination.device
        )
        caches = list(ordinary_caches[: self.destination_layer + 1])
        hidden_states = mixed_destination
        for layer_index in range(self.destination_layer + 1, len(self.backbone.model.layers)):
            layer = self.backbone.model.layers[layer_index]
            past = None if previous_caches is None else previous_caches[layer_index]
            hidden_states, cache = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past,
                use_cache=True,
                detach_cache=detach_cache,
                fast_attention_compatible=False,
            )
            if cache is None:
                raise RuntimeError("recirculation replay did not return KV state")
            caches.append(cache)
        return tuple(caches)

    def _recirculate_token(
        self,
        token: torch.Tensor,
        previous_caches: tuple[LayerKVCache, ...] | None,
        *,
        detach_cache: bool,
        full_replay: bool,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        """Read out one token, then replay it into the cache per Eq. 1."""
        ordinary = self._run_core(
            self.input_embeddings(token),
            recurrent_source=None,
            past_key_values=previous_caches,
            use_cache=True,
            detach_cache=detach_cache,
        )
        if ordinary.past_key_values is None:
            raise RuntimeError("paper recirculation first iteration returned no cache")
        mixed_destination = self._mix(
            ordinary.source_state,
            ordinary.destination_state,
        )
        if full_replay:
            replay = self._run_core(
                self.input_embeddings(token),
                recurrent_source=None,
                destination_override=mixed_destination,
                past_key_values=previous_caches,
                use_cache=True,
                detach_cache=detach_cache,
            )
            if replay.past_key_values is None:
                raise RuntimeError("paper recirculation replay returned no cache")
            replay_caches = replay.past_key_values
        else:
            replay_caches = self._replay_upper_stack(
                mixed_destination,
                ordinary_caches=ordinary.past_key_values,
                previous_caches=previous_caches,
                detach_cache=detach_cache,
            )
        logits = self.backbone.lm_head(ordinary.hidden_states).float()
        return logits, replay_caches

    def _training_recirculation_token(
        self,
        token: torch.Tensor,
        caches: tuple[LayerKVCache, ...] | None,
        *,
        activation_checkpointing: bool,
        full_replay: bool = False,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        """Run one differentiable token step with optional recomputation."""
        if activation_checkpointing and torch.is_grad_enabled():
            previous_caches = caches

            def token_step(
                current_token: torch.Tensor,
                previous_caches=previous_caches,
            ):
                return self._recirculate_token(
                    current_token,
                    previous_caches,
                    detach_cache=False,
                    full_replay=full_replay,
                )

            return checkpoint(token_step, token, use_reentrant=False)
        return self._recirculate_token(
            token,
            caches,
            detach_cache=not torch.is_grad_enabled(),
            full_replay=full_replay,
        )

    def compute_recirculation_logits(
        self,
        input_ids: torch.Tensor,
        *,
        activation_checkpointing: bool = False,
        full_replay: bool = False,
    ) -> torch.Tensor:
        """Teacher-force the paper's token-diagonal recirculation forward.

        Each position is read out after its ordinary first iteration. The same
        token is then replayed from the source/destination mixture, replacing
        its upper-layer K/V entries before the next token is processed. With
        autograd enabled, caches remain attached so loss gradients propagate
        through the complete token chain (BPTT).
        """
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("input_ids must be [B,T] with at least two tokens")
        caches: tuple[LayerKVCache, ...] | None = None
        logits_by_token: list[torch.Tensor] = []
        for position in range(input_ids.shape[1]):
            token = input_ids[:, position : position + 1]
            logits, caches = self._training_recirculation_token(
                token,
                caches,
                activation_checkpointing=activation_checkpointing,
                full_replay=full_replay,
            )
            logits_by_token.append(logits)
        return torch.cat(logits_by_token, dim=1)

    def compute_recirculation_bptt_loss(
        self,
        input_ids: torch.Tensor,
        *,
        activation_checkpointing: bool = False,
    ) -> TrainOutput:
        """Next-token loss through the complete paper-recirculation chain."""
        logits = self.compute_recirculation_logits(
            input_ids,
            activation_checkpointing=activation_checkpointing,
        )
        loss = self.lm_loss(logits, input_ids)
        metrics = self.recirculation_compute_metrics()
        metrics["recirculation_bptt_loss"] = float(loss.detach().cpu())
        return TrainOutput(
            loss=loss,
            pass_losses=(loss,),
            effective_passes=2,
            metrics=metrics,
        )

    def recirculation_compute_metrics(self) -> dict[str, float]:
        """Describe the partial replay behind the two-iteration accounting."""
        total_layers = len(self.backbone.model.layers)
        replayed_layers = total_layers - self.destination_layer - 1
        return {
            "recirculation_stack_iterations": 2.0,
            "recirculation_replayed_layers": float(replayed_layers),
            "recirculation_total_layers": float(total_layers),
            "recirculation_replay_layer_fraction": replayed_layers / total_layers,
        }

    @staticmethod
    def _detach_recirculation_caches(
        caches: tuple[LayerKVCache, ...],
    ) -> tuple[LayerKVCache, ...]:
        """Preserve recurrent values while cutting gradients at a TBPTT boundary."""
        return tuple(
            LayerKVCache(
                key=cache.key.detach(),
                value=cache.value.detach(),
                start_pos=cache.start_pos,
                key_valid=cache.key_valid,
                positions=cache.positions,
                next_pos=cache.next_pos,
            )
            for cache in caches
        )

    def iter_recirculation_tbptt_losses(
        self,
        input_ids: torch.Tensor,
        *,
        truncate_tokens: int,
        activation_checkpointing: bool = False,
    ) -> Iterable[tuple[torch.Tensor, int]]:
        """Yield token-weighted loss chunks with detached recurrent boundaries.

        The forward recurrence and KV values continue across the whole packed
        sequence. Only the gradient path is cut every ``truncate_tokens`` input
        positions. The caller must backpropagate each yielded loss before
        requesting the next chunk; this is what releases the completed graph
        instead of retaining every chunk until a final backward call.
        """
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("input_ids must be [B,T] with at least two tokens")
        if truncate_tokens < 2:
            raise ValueError("truncate_tokens must be at least 2")

        caches: tuple[LayerKVCache, ...] | None = None
        sequence_length = int(input_ids.shape[1])
        for start in range(0, sequence_length, int(truncate_tokens)):
            stop = min(start + int(truncate_tokens), sequence_length)
            logits_by_token: list[torch.Tensor] = []
            for position in range(start, stop):
                token = input_ids[:, position : position + 1]
                logits, caches = self._training_recirculation_token(
                    token,
                    caches,
                    activation_checkpointing=activation_checkpointing,
                )
                logits_by_token.append(logits)

            prediction_positions = min(stop, sequence_length - 1) - start
            if prediction_positions:
                logits = torch.cat(logits_by_token[:prediction_positions], dim=1)
                targets = input_ids[:, start + 1 : start + 1 + prediction_positions]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
                yield loss, int(targets.numel())
            if caches is None:
                raise RuntimeError("recirculation TBPTT chunk returned no cache")
            caches = self._detach_recirculation_caches(caches)

    def compute_training_loss(
        self,
        input_ids: torch.Tensor,
        *,
        training_forward: str,
        phase: str,
        passes: int,
        loss_weights: Sequence[float] | None,
        activation_checkpointing: bool = False,
    ) -> TrainOutput:
        if training_forward == "recirculation_bptt":
            if passes != 1 or loss_weights is not None:
                raise ValueError(
                    "recirculation_bptt uses one diagonal recurrence policy, not K/pass weights"
                )
            return self.compute_recirculation_bptt_loss(
                input_ids,
                activation_checkpointing=activation_checkpointing,
            )
        return super().compute_training_loss(
            input_ids,
            training_forward=training_forward,
            phase=phase,
            passes=passes,
            loss_weights=loss_weights,
            activation_checkpointing=activation_checkpointing,
        )

    def _run_first_state(self, input_ids: torch.Tensor) -> HiddenRun:
        return self._core(
            self.input_embeddings(input_ids),
            recurrent_source=None,
            past_key_values=None,
            use_cache=False,
        )

    def _run_feedback_state(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_source: torch.Tensor,
    ) -> HiddenRun:
        del input_ids
        return self._core(
            token_embeddings,
            recurrent_source=shift_previous_hidden(previous_source),
            past_key_values=None,
            use_cache=False,
            full_sequence_feedback=True,
        )

    def _run_first_state_cached(self, input_ids: torch.Tensor) -> HiddenRun:
        return self._core(
            self.input_embeddings(input_ids),
            recurrent_source=None,
            past_key_values=None,
            use_cache=True,
        )

    def _run_feedback_state_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_source: torch.Tensor,
    ) -> HiddenRun:
        del input_ids
        return self._core(
            token_embeddings,
            recurrent_source=shift_previous_hidden(previous_source),
            past_key_values=None,
            use_cache=True,
            full_sequence_feedback=True,
        )

    def _run_first_token_state_cached(
        self,
        input_ids: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
    ) -> HiddenRun:
        return self._core(
            self.input_embeddings(input_ids),
            recurrent_source=None,
            past_key_values=past_key_values,
            use_cache=True,
        )

    def _run_feedback_token_state_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> HiddenRun:
        del token
        return self._core(
            token_embedding,
            recurrent_source=feedback_memory,
            past_key_values=past_key_values,
            use_cache=True,
        )

    # Legacy hidden-only hooks remain available to callers of the original
    # multipass API; the richer state hooks above carry the source layer.
    def _run_feedback_hidden(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> torch.Tensor:
        return self._run_feedback_state(input_ids, token_embeddings, previous_hidden).hidden_states

    def _run_feedback_hidden_cached(
        self,
        input_ids: torch.Tensor,
        token_embeddings: torch.Tensor,
        previous_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        run = self._run_feedback_state_cached(input_ids, token_embeddings, previous_hidden)
        if run.past_key_values is None:
            raise RuntimeError("cached recirculation pass did not return KV state")
        return run.hidden_states, run.past_key_values

    def _run_feedback_token_cached(
        self,
        token_embedding: torch.Tensor,
        feedback_memory: torch.Tensor,
        past_key_values: tuple[LayerKVCache, ...],
        *,
        token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[LayerKVCache, ...]]:
        run = self._run_feedback_token_state_cached(
            token_embedding,
            feedback_memory,
            past_key_values,
            token=token,
        )
        if run.past_key_values is None:
            raise RuntimeError("cached recirculation token did not return KV state")
        return run.hidden_states, run.past_key_values

    def _feedback_memory_from_hidden(
        self,
        feedback_source: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids
        if feedback_source.ndim != 3 or feedback_source.shape[1] < 1:
            raise ValueError("feedback_source must be non-empty [B,T,D]")
        return feedback_source[:, -1:, :].detach()

    def _append_feedback_memory(
        self,
        feedback_memory: torch.Tensor,
        new_feedback_source: torch.Tensor,
        *,
        token: torch.Tensor | None = None,
        position: int | None = None,
    ) -> torch.Tensor:
        del token, position
        if feedback_memory.ndim != 3 or feedback_memory.shape[1] != 1:
            raise ValueError("recirculation feedback memory must be [B,1,D]")
        if new_feedback_source.ndim != 3 or new_feedback_source.shape[1] != 1:
            raise ValueError("new recirculation source must be [B,1,D]")
        return new_feedback_source.detach()
