# Cached and feedback inference

This document defines cached inference for `recurrent_memory`, Memory Attention,
and supported legacy multipass variants. Prompt
refinement depth K is an inference-time
parameter and need not equal the training depth.

Whole-block multipass refinement has a prompt-depth K. The paper's same-token
readout/replay execution has been deleted. The adaptive recirculation merger
remains an option within the preceding-token memory architecture.

## Exact incremental K

`exact_incremental` keeps K independent TinyMistral self-attention KV streams.
Stream 1 is the first-pass stream. Stream `k>1` consumes strict-past feedback
from stream `k-1` using the architecture-specific routing rule. Recirculation
stores its source-layer state, so cached and full-sequence execution use the
same layer and shift.

The central invariant is **snapshot before update**. At physical position t,
every higher stream reads the lower-stream feedback state that existed before t.
New `h_t` states are appended to feedback memories only after all K streams
finish the position. This prevents same-position recurrence leakage.

The implementation is tested against full-prefix `compute_passes(...,
passes=K)` recomputation for multiple K values and all Memory Attention write/visibility
modes.

## Prompt prefill and continuation decode

Prompt pass depth and continuation mechanism are independent axes.
`prefill_passes=K` always describes how many complete prompt passes construct
the initial state. `decode_mode="standard"` then advances an ordinary cached
stream. `decode_mode="feedback"` advances the live final stream once per
observed or generated token and feeds each new state back into the next step.

For K>1 feedback decoding pairs the final-pass cache with feedback from pass
K-1. The first continuation transition therefore agrees with exact K-pass
inference. For K=1, feedback decoding still constructs and uses the
architecture-specific prompt feedback memory; it is not silently disabled.
Vanilla models reject feedback mode.

For long-form generation experiments, K applies only to prompt prefill. The
continuation must use `decode_mode="feedback"`; it must not keep rerunning K
complete streams for every generated token, and it must not silently fall back
to standard decoding. Candidate-loglikelihood tasks consume each observed
candidate token through the same live feedback state.

Paper readout/replay inference has been deleted, not renamed to feedback.
BOS-only initialization is the ordinary `feedback` mode with a one-token BOS
prompt and `prefill_passes=1`; context-prefilled downstream use defaults to K=4.
The packed-text evaluator and selected-checkpoint schedule reuse this path.

## Memory Attention state

Cached Memory Attention feedback is a fixed-capacity chronological `MemoryAttentionState` with at most
`memory_window` records and an explicit validity mask. Decode obeys:

```text
read old memory -> compute token hidden -> optionally append writer(hidden)
```

so the current position cannot read its own write. When the memory is full, a
triggered append evicts the oldest record.

Dense writes every ordinary position. Strided writes use the absolute physical
position and configured stride. Memory-token mode writes only when the observed
input token is ID V.

Dense-and-strided Memory Attention also writes every ordinary position, but its bounded state is
the chronological union of the last `D` positions and the last `S` older
fixed-stride positions. Every append recomputes retention relative to the
next query coordinate while preserving cached per-reader projected K/V.

## Optional hybrid state

The general late-memory hybrid stores `HybridFeedbackState` with
`recurrent_memory` (the preceding ordinary token's emitted record) and
`memory_attention` (the bounded attention state). Both use one shared writer.
A MEM step may append an attention record but preserves `recurrent_memory`.
An ordinary step updates recurrence even when no attention write is due.
Exact K=1 conversion preserves both channels without applying the writer twice.

## Memory-token decode

MEM is a physical control position, not a predicted language token. Consuming a
MEM position therefore keeps `next_token_logits` from the preceding ordinary
position. The intended free-running schedule is:

```text
ordinary hidden predicts next linguistic token B
if a MEM slot is due: process MEM internally and write its memory state
process already selected B
B hidden predicts the next linguistic token
```

The low-level API consumes explicit observed physical tokens; public
`MultiPassVariant.generate()` remains the ordinary backbone generator rather
than silently inserting control positions.

In `write_only` mode every self-attention layer cache carries a boolean
`key_valid` mask. A MEM K/V entry remains in the cache at its physical/RoPE
position but has validity false, so later cached queries cannot self-attend to
it.

## API

```python
from tiny_mistral_mptt.inference import prefill, decode_step

state = prefill(model, input_ids, passes=K, mode="exact_incremental")
state = decode_step(model, state, observed_token)

state = prefill(
    model, input_ids, passes=K,
    mode="recurrent", decode_mode="feedback",
)
state = decode_step(model, state, observed_token)


```

Dedicated `prefill_exact`, `prefill_recurrent`, `exact_decode_step`, and
`recurrent_decode_step` helpers are also exported. State objects are immutable;
decode returns a new state.

## Correctness and evaluation

See [validation gates](VALIDATION.md) for cache equivalence, strict-past routing,
bounded memory and MEM visibility checks.

Held-out training-distribution NLL and downstream generation answer different
questions. Whole-block NLL reports every parallel pass through the configured
depth (K=4 in the active frozen studies). Continuation diagnostics compare exact
K-stream decoding, ordinary feedback, and a standard K=1 stream from the same
checkpoint. They do not implement paper replay. Downstream generation uses the
selected prompt prefill followed by live feedback for the continuation.

Full-block BOS feedback is available through `evaluate_nll.py --forward feedback`
and selected-checkpoint trainer validation. It uses `prefill_recurrent` with
one BOS token, `passes=1`, `decode_mode="feedback"`, then teacher-forces every
data target. Full and aligned scores distinguish 2048 targets from the ordinary
2047-target next-token comparison. See [evaluation](../evaluation/README.md).

## Compatibility

Exact K=1 decoding updates architecture-specific feedback memory after each token.
Conversion to feedback therefore preserves a learned, non-identity writer.
See [evaluation](../evaluation/README.md) for result keys and historical JSON compatibility.
