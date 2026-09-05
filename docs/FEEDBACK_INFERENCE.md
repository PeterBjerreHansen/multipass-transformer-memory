# Exact K-pass and Live Feedback inference

Prompt refinement depth and continuation policy are independent. The public
runtime names are:

- `exact_k_pass`: retain K cached pass streams and reproduce full-prefix
  `compute_passes(..., passes=K)` as tokens are appended;
- `live_feedback`: collapse an exact prompt prefill to one stream and feed each
  newly produced state into the next temporal step; and
- `standard_k1`: ordinary cached decoding from a one-pass prompt state.

`MultiPassVariant.forward()` and `.generate()` intentionally raise because
neither method can choose among these semantics without changing the estimand.
Use `compute_passes`, `prefill_exact_k_pass`, `prefill_live_feedback`,
`exact_decode_step`, or `live_feedback_decode_step` explicitly. The LM-evaluation
adapter makes the same choice from its `prefill_passes` and `decode_mode`
arguments.

## Exact K-pass invariant

At physical position `t`, every higher stream reads the lower-stream feedback
state that existed before `t`. New states are appended only after all K streams
finish the position. This snapshot-before-update rule prevents same-position
leakage and is tested against full-prefix recomputation.

## Live Feedback handoff

For a K-pass prompt with K>1, Live Feedback pairs the final-pass self-attention
cache with feedback from pass K-1. Its initial prediction and first processed
continuation transition therefore agree with exact K-pass inference. Later
tokens close the loop over the observed or generated continuation. K=1 still
uses the architecture's feedback pathway; it does not silently become standard
decoding.

## Memory state

Dense Memory Attention writes every physical position. Strided Memory Attention
writes zero-based physical position `t` when `(t + 1) % C == 0`. A synthetic BOS
is an ordinary physical position and shifts the subsequent data-token phase.
Memory-token mode writes only explicit control positions. All reads are strict
past, and cached memory remains bounded by its configured capacity.

The separate continuation diagnostic reports exact-K-pass, Live Feedback, and
standard-K1 NLL; `KL(exact || live_feedback)`; top-1 agreement; hidden RMS and
cosine drift; and retained K=4 improvement. The intervention diagnostic tests
real, zero, mismatched, and true-bypass conditions at configurable transitions.
Neither diagnostic is routine validation or a training stopping criterion.

Old Python runtime aliases are not exported. Compatibility remains only where
serialized historical artifacts require it; new code, commands, and result
schemas use the explicit names above.
