# Exact incremental and collapsed recurrent inference

This document defines cached inference for the active `recirculation`,
`memory_attention`/`bank`, and `memory_attention_multiscale`/`bank_multiscale`
variants. Prompt refinement depth K is an inference-time
parameter and need not equal the training depth.

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

## Collapsed recurrent K

`recurrent` performs the same exact K-pass prompt prefill, then retains only the
final-pass self-attention cache and, for K>1, the feedback state from pass K-1.
The first processed continuation position therefore sees exactly the same
feedback and final-stream history as exact K-pass inference. After that position,
the live final stream writes its own new feedback and closes the recurrent loop.

K=1 is the vanilla cached boundary: recurrent feedback is disabled.

## Memory Attention state

Cached Memory Attention feedback is a fixed-capacity chronological `BankState` with at most
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

state = prefill(model, input_ids, passes=K, mode="recurrent")
state = decode_step(model, state, observed_token)
```

Dedicated `prefill_exact`, `prefill_recurrent`, `exact_decode_step`, and
`recurrent_decode_step` helpers are also exported. State objects are immutable;
decode returns a new state.

## Required gates

Before interpreting recurrent quality:

- exact cached K-pass must match full-prefix recomputation;
- K=1 must reduce to ordinary cached TinyMistral;
- the first collapsed recurrent transition must match exact K-pass;
- Memory Attention state must remain bounded and strict-past;
- multiscale Memory Attention state must retain the exact sparse-old/dense-recent union;
- write-only MEM validity must survive KV caching without changing physical
  positions;
- cached absolute positions must remain correct beyond the self-attention
  sliding window.
