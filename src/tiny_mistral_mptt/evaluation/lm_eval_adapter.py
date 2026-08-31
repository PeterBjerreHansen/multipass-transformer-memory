from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ..inference import (
    DecodeMode,
    paper_recirculation_decode_step,
    prefill_paper_recirculation,
    prefill_recurrent,
    recurrent_decode_step,
)
from ..variants.multipass import MultiPassVariant
from ..variants.recirculation import RecirculationVariant


try:  # Optional dependency: imported only for the external benchmark battery.
    from lm_eval import utils as lm_eval_utils
    from lm_eval.api.model import TemplateLM
except ImportError:  # pragma: no cover - environment dependent
    TemplateLM = None  # type: ignore[assignment]
    lm_eval_utils = None  # type: ignore[assignment]




@torch.no_grad()
def score_token_continuation(
    model,
    *,
    device: str | torch.device,
    max_length: int,
    context_enc: list[int],
    continuation_enc: list[int],
) -> tuple[float, bool]:
    """Score one already-tokenized causal context/continuation pair.

    This is intentionally independent of lm-eval so the indexing contract can
    be unit-tested even when the optional harness package is unavailable.
    """
    if not context_enc:
        raise ValueError("context_enc must contain at least one prefix/context token")
    if not continuation_enc:
        return 0.0, True
    combined = list(context_enc) + list(continuation_enc)
    if len(combined) > max_length + 1:
        removed = len(combined) - (max_length + 1)
        combined = combined[removed:]
        remaining_context = max(len(context_enc) - removed, 0)
    else:
        remaining_context = len(context_enc)
    if len(combined) < 2:
        return 0.0, True
    input_tokens = combined[:-1]
    target_tokens = combined[1:]
    start = max(remaining_context - 1, 0)
    scored_targets = target_tokens[start:]
    if not scored_targets:
        return 0.0, True
    ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
    output = model(ids, use_cache=False)
    logits = output.logits[:, start : start + len(scored_targets), :].float()
    targets = torch.tensor([scored_targets], dtype=torch.long, device=device)
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    total = float(chosen.sum().cpu())
    greedy = bool(torch.equal(logits.argmax(dim=-1), targets))
    return total, greedy


def _truncate_token_pair(
    *,
    max_length: int,
    context_enc: list[int],
    continuation_enc: list[int],
) -> tuple[list[int], list[int]]:
    """Return a cached prompt and the continuation targets to score.

    The non-cached harness path scores ``combined[:-1]`` against
    ``combined[1:]``.  Recurrent inference needs the same causal boundary, but
    can only prefill a non-empty prompt and then consume targets one at a time.
    ``prompt`` therefore ends immediately before the first scored target.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if not context_enc:
        raise ValueError("context_enc must contain at least one prefix/context token")
    if not continuation_enc:
        return list(context_enc[-max_length:]), []

    combined = list(context_enc) + list(continuation_enc)
    if len(combined) > max_length + 1:
        removed = len(combined) - (max_length + 1)
        combined = combined[removed:]
        remaining_context = max(len(context_enc) - removed, 0)
    else:
        remaining_context = len(context_enc)

    input_tokens = combined[:-1]
    target_tokens = combined[1:]
    start = max(remaining_context - 1, 0)
    if not target_tokens[start:]:
        return input_tokens, []
    prompt = combined[: start + 1]
    if not prompt:
        raise ValueError("recurrent scoring requires a non-empty prompt")
    return prompt, target_tokens[start:]


@torch.no_grad()
def score_token_continuation_recurrent(
    model: MultiPassVariant,
    *,
    device: str | torch.device,
    max_length: int,
    prefill_passes: int,
    decode_mode: DecodeMode,
    context_enc: list[int],
    continuation_enc: list[int],
) -> tuple[float, bool]:
    """Score a continuation with recurrent cached inference.

    The prompt is refined ``prefill_passes`` times.  Each continuation target
    is scored from the current cached logits, then observed once through
    ``recurrent_decode_step`` to advance the one-stream state.  The final
    target is not decoded because no later prediction depends on it, which
    keeps the observed sequence within the model position limit.
    """
    if prefill_passes < 1:
        raise ValueError("prefill_passes must be positive")
    prompt, targets = _truncate_token_pair(
        max_length=max_length,
        context_enc=context_enc,
        continuation_enc=continuation_enc,
    )
    if not targets:
        return 0.0, True

    prompt_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    if decode_mode == "paper_recirculation":
        if not isinstance(model, RecirculationVariant):
            raise ValueError("paper_recirculation requires RecirculationVariant")
        if prefill_passes != 1:
            raise ValueError("paper_recirculation has no prompt K axis; use prefill_passes=1")
        state = prefill_paper_recirculation(model, prompt_ids)
    else:
        state = prefill_recurrent(
            model,
            prompt_ids,
            passes=prefill_passes,
            decode_mode=decode_mode,
        )
    total = 0.0
    greedy = True
    for index, target_id in enumerate(targets):
        target = torch.tensor([[target_id]], dtype=torch.long, device=device)
        logits = state.next_token_logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        total += float(log_probs[0, target_id].cpu())
        greedy = greedy and bool(torch.argmax(logits, dim=-1).eq(target[:, 0]).item())

        # The final target has already been scored and does not need to be
        # consumed.  Avoiding that extra cache write preserves max_length + 1
        # harness scoring semantics.
        if index + 1 < len(targets):
            if decode_mode == "paper_recirculation":
                assert isinstance(model, RecirculationVariant)
                state = paper_recirculation_decode_step(model, state, target)
            else:
                state = recurrent_decode_step(model, state, target)
    return total, greedy


@torch.no_grad()
def generate_recurrent(
    model: MultiPassVariant,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    prefill_passes: int,
    decode_mode: DecodeMode,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate with one recurrent cached stream after multipass prefill."""
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must be non-empty [B,T]")
    if input_ids.shape[0] != 1:
        raise ValueError("recurrent generation currently supports batch size 1")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")
    if max_new_tokens == 0:
        return input_ids

    if decode_mode == "paper_recirculation":
        if not isinstance(model, RecirculationVariant):
            raise ValueError("paper_recirculation requires RecirculationVariant")
        if prefill_passes != 1:
            raise ValueError("paper_recirculation has no prompt K axis; use prefill_passes=1")
        state = prefill_paper_recirculation(model, input_ids)
    else:
        state = prefill_recurrent(
            model,
            input_ids,
            passes=prefill_passes,
            decode_mode=decode_mode,
        )
    eos = model.config.eos_token_id if eos_token_id is None else eos_token_id
    result = input_ids
    for step in range(max_new_tokens):
        logits = state.next_token_logits.float()
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            if top_k is not None:
                k = min(top_k, scaled.shape[-1])
                threshold = torch.topk(scaled, k, dim=-1).values[:, -1:]
                scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
            next_token = torch.multinomial(F.softmax(scaled, dim=-1), 1)
        result = torch.cat((result, next_token), dim=1)
        if eos is not None and bool(torch.all(next_token.squeeze(-1) == eos).item()):
            break
        if step == max_new_tokens - 1:
            break
        if decode_mode == "paper_recirculation":
            assert isinstance(model, RecirculationVariant)
            state = paper_recirculation_decode_step(model, state, next_token)
        else:
            state = recurrent_decode_step(model, state, next_token)
    return result


class _TokenizerFacade:
    def __init__(self, path: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("evaluation requires: uv sync --extra eval") from exc
        self.inner = Tokenizer.from_file(str(path))
        # The published tokenizer.json carries fixed-length padding and
        # truncation for its original training pipeline. lm-eval requires raw,
        # variable-length encodings so it can identify the exact continuation
        # boundary itself.
        self.inner.no_padding()
        self.inner.no_truncation()

    def encode(self, text: str) -> list[int]:
        return self.inner.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.inner.decode(ids, skip_special_tokens=False)


def _build_lm_eval_class():
    if TemplateLM is None:  # pragma: no cover - optional dependency
        return None

    class TinyMistralHarnessLM(TemplateLM):
        """Minimal single-process lm-evaluation-harness adapter for this repo."""

        backend = "causal"

        def __init__(
            self,
            model,
            *,
            tokenizer_path: str | Path,
            device: str | torch.device,
            max_gen_toks: int = 128,
            decode_mode: DecodeMode,
            prefill_passes: int,
        ):
            super().__init__()
            if decode_mode not in {"standard", "feedback", "paper_recirculation"}:
                raise ValueError(
                    "decode_mode must be standard, feedback, or paper_recirculation"
                )
            if prefill_passes < 1:
                raise ValueError("prefill_passes must be positive")
            if not isinstance(model, MultiPassVariant):
                if prefill_passes != 1:
                    raise ValueError("vanilla models require prefill_passes=1")
                if decode_mode == "feedback":
                    raise ValueError("vanilla models do not implement feedback decoding")
                if decode_mode == "paper_recirculation":
                    raise ValueError("vanilla models do not implement recurrent decoding")
            if decode_mode == "paper_recirculation":
                if not isinstance(model, RecirculationVariant):
                    raise ValueError("paper_recirculation requires RecirculationVariant")
                if prefill_passes != 1:
                    raise ValueError(
                        "paper_recirculation has no prompt K axis; use prefill_passes=1"
                    )
            self.model = model
            self._device = torch.device(device)
            self._tokenizer_facade = _TokenizerFacade(tokenizer_path)
            self._max_gen_toks = int(max_gen_toks)
            self._decode_mode = decode_mode
            self._prefill_passes = int(prefill_passes)
            self.batch_size = 1
            self.model.eval()

        @property
        def eot_token_id(self) -> int:
            return int(self.model.config.eos_token_id)

        @property
        def prefix_token_id(self) -> int:
            return int(self.model.config.bos_token_id)

        @property
        def max_length(self) -> int:
            return int(self.model.config.max_position_embeddings)

        @property
        def max_gen_toks(self) -> int:
            return self._max_gen_toks

        @property
        def tokenizer_name(self) -> str:
            return "TinyMistral-248M-v3-tokenizer"

        @property
        def decode_mode(self) -> str:
            return self._decode_mode

        @property
        def prefill_passes(self) -> int:
            return self._prefill_passes

        def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
            # TinyMistral's baseline experiments use raw tokenizer.json encoding;
            # BOS is inserted explicitly only when the harness needs an empty context.
            return self._tokenizer_facade.encode(string)

        def tok_decode(self, tokens: list[int]) -> str:
            return self._tokenizer_facade.decode([int(token) for token in tokens])

        @torch.no_grad()
        def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
            results: list[tuple[float, bool]] = []
            for cache_key, context_enc, continuation_enc in requests:
                if isinstance(self.model, MultiPassVariant):
                    answer = score_token_continuation_recurrent(
                        self.model,
                        device=self._device,
                        max_length=self.max_length,
                        prefill_passes=self._prefill_passes,
                        decode_mode=self._decode_mode,
                        context_enc=list(context_enc),
                        continuation_enc=list(continuation_enc),
                    )
                else:
                    answer = score_token_continuation(
                        self.model,
                        device=self._device,
                        max_length=self.max_length,
                        context_enc=list(context_enc),
                        continuation_enc=list(continuation_enc),
                    )
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
            return results

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
            if lm_eval_utils is None:
                raise RuntimeError("lm-eval is unavailable")
            results: list[float] = []
            for request in requests:
                string = request.args[0]
                tokens = self.tok_encode(string)
                windows = list(
                    map(
                        lm_eval_utils.make_disjoint_window,
                        lm_eval_utils.get_rolling_token_windows(
                            token_list=tokens,
                            prefix_token=self.prefix_token_id,
                            max_seq_len=self.max_length,
                            context_len=1,
                        ),
                    )
                )
                token_requests = [(None,) + window for window in windows]
                scored = self._loglikelihood_tokens(token_requests, disable_tqdm=disable_tqdm)
                total = sum(item[0] for item in scored)
                results.append(total)
                self.cache_hook.add_partial("loglikelihood_rolling", (string,), total)
            return results

        @torch.no_grad()
        def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
            results: list[str] = []
            for request in requests:
                context, gen_kwargs = request.args
                gen_kwargs = dict(gen_kwargs or {})
                until = gen_kwargs.get("until", [])
                if isinstance(until, str):
                    until = [until]
                max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
                temperature = float(gen_kwargs.get("temperature", 0.0))
                top_k = gen_kwargs.get("top_k")
                context_ids = self.tok_encode(context)
                if not context_ids:
                    context_ids = [self.prefix_token_id]
                # Leave room for requested generation inside the vanilla RoPE limit.
                max_prompt = max(1, self.max_length - max_new)
                context_ids = context_ids[-max_prompt:]
                prompt = torch.tensor([context_ids], dtype=torch.long, device=self._device)
                if isinstance(self.model, MultiPassVariant):
                    generated = generate_recurrent(
                        self.model,
                        prompt,
                        max_new,
                        prefill_passes=self._prefill_passes,
                        decode_mode=self._decode_mode,
                        temperature=temperature,
                        top_k=None if top_k is None else int(top_k),
                    )
                else:
                    generated = self.model.generate(
                        prompt,
                        max_new,
                        temperature=temperature,
                        top_k=None if top_k is None else int(top_k),
                    )
                suffix_ids = generated[0, prompt.shape[1] :].tolist()
                text = self.tok_decode(suffix_ids)
                cut = len(text)
                for stop in until:
                    position = text.find(stop)
                    if position >= 0:
                        cut = min(cut, position)
                text = text[:cut]
                results.append(text)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
            return results

    return TinyMistralHarnessLM


TinyMistralHarnessLM = _build_lm_eval_class()


def make_lm_eval_adapter(
    model,
    *,
    tokenizer_path: str | Path,
    device: str | torch.device,
    max_gen_toks: int = 128,
    decode_mode: DecodeMode,
    prefill_passes: int,
):
    if TinyMistralHarnessLM is None:
        raise RuntimeError("lm-evaluation-harness is not installed; run: uv sync --extra eval")
    return TinyMistralHarnessLM(
        model,
        tokenizer_path=tokenizer_path,
        device=device,
        max_gen_toks=max_gen_toks,
        decode_mode=decode_mode,
        prefill_passes=prefill_passes,
    )
