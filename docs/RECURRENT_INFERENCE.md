# Exact incremental and collapsed recurrent inference

This document defines cached inference for the active `recirculation`,
`memory_attention` and `multiscale_memory_attention` variants. Prompt
refinement depth K is an inference-time
parameter and need not equal the training depth.

There are two distinct recirculation semantics in the repository. Whole-block
multipass refinement has a prompt-depth K. Paper recirculation instead performs
one ordinary readout followed by one same-token replay at every position; it has
no K axis.

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

`decode_mode="paper_recirculation"` is the corresponding end-to-end mode for a
Recirculation model trained with token-diagonal BPTT. Prompt tokens and generated
tokens both use the paper's readout-then-replay update, so
`prefill_passes` must be 1. The value is retained only for CLI/schema
compatibility and does not introduce a prompt K.

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

Multiscale Memory Attention also writes every ordinary position, but its bounded state is
the chronological union of the last `D` positions and the last `S` older
fixed-stride positions. Every append recomputes retention relative to the
next query coordinate while preserving cached per-reader projected K/V.

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

from tiny_mistral_mptt.inference import (
    paper_recirculation_decode_step,
    prefill_paper_recirculation,
)

paper_state = prefill_paper_recirculation(recirculation_model, input_ids)
paper_state = paper_recirculation_decode_step(
    recirculation_model, paper_state, observed_token
)
```

Dedicated `prefill_exact`, `prefill_recurrent`, `exact_decode_step`, and
`recurrent_decode_step` helpers are also exported. State objects are immutable;
decode returns a new state.

## Required gates

Before interpreting recurrent quality:

- exact cached K-pass must match full-prefix recomputation;
- exact K=1 and K=1 standard decode must reduce to cached TinyMistral;
- K=1 feedback decode must retain a real feedback state and remain distinct
  from standard decode;
- the first collapsed recurrent transition must match exact K-pass;
- Memory Attention state must remain bounded and strict-past;
- multiscale Memory Attention state must retain the exact sparse-old/dense-recent union;
- write-only MEM validity must survive KV caching without changing physical
  positions;
- cached absolute positions must remain correct beyond the self-attention
  sliding window.

Held-out training-distribution NLL and downstream generation answer different
questions. Whole-block NLL at K=1 through K=8 is a convergence diagnostic for
the parallel forward. Recurrent teacher-forced NLL measures the paper forward
and has no K. Downstream generation uses the selected prompt prefill followed
by the live feedback mechanism for the continuation.
