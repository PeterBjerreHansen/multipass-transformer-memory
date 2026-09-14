# Architecture contracts

This map separates active model families and supported controls.
Study settings belong in the [benchmark protocols](../benchmarks/README.md).
The detailed contracts are [recurrent memory](RECURRENT_MEMORY.md) and
[Memory Attention](MEMORY_ATTENTION.md).

## SWA Transformer

One ordinary TinyMistral causal pass with no architecture-added parameters.

## Strided Self-Attention control

`StridedSelfAttentionVariant` is a one-pass, parameter-free Transformer control. At the
selected `sparse_attention_layers`, the existing Mistral self-attention uses
one softmax over the union of its ordinary SWA keys and a bounded set of older
fixed-stride keys. It reuses the pretrained Q/K/V/O projections and does not
add a Memory Attention reader or any cross-attention parameters.

For query `t`, SWA width `W`, sparse stride `C`, and sparse count `S`, the added
keys are the last `S` positions satisfying `(s + 1) % C == 0` and `s < t-W+1`.
The sparse region never duplicates a key in the local region. Cached decoding
retains `W-1` recent K/V entries plus at most `S` older strided entries with
their absolute RoPE positions.

```yaml
variant: strided_self_attention
sparse_attention_stride: 32
sparse_attention_window: 32
sparse_attention_layers: [3, 7]
```

## No-memory Adapter

`NoMemoryAdapterVariant` is the capacity control for the active frozen study.
Pass 1 is the unchanged backbone. On later passes, the first selected site
projects its current residual through the same shared D-to-D writer shape used
by projected-residual fixed-route feedback. The resulting within-pass control
record is reused by the projected-residual merger at every selected site. It
never reads an earlier-pass or earlier-token Feedback Record.

This construction exactly matches the projected-residual arm's added-parameter
count at a fixed site count while testing whether feedback transfer adds value
beyond trainable later-pass capacity. Real, zero, and mismatched memory inputs
must therefore give it identical outputs; true bypass remains different because
it omits the adapter path itself.

## Recurrent memory

`RecurrentMemoryVariant` uses the same late writer as Memory Attention, reads
only the preceding token's previous-pass memory, and selects one of two mergers
with `recurrent_merger: projected_residual|recirculation`. Reads occur after
self-attention and before the MLP at `memory_layers`. Use `memory_window: 1`; study configs select the reader layout. See
[RECURRENT_MEMORY.md](RECURRENT_MEMORY.md) for initialization, gradient flow,
comparison limits, and inference semantics.

## Memory Attention

`dense_memory_attention`, `strided_memory_attention` and
`dense_and_strided_memory_attention` are presets of `memory_attention`, using
one model implementation. They share a late writer and selected
post-self-attention readers. See [Memory Attention](MEMORY_ATTENTION.md) for
masks, retention, initialization, gradient flow and aliases.

### Optional recurrent-memory hybrid

Set `recurrent_merger` and explicit `recurrent_layers` on a Memory Attention
configuration. Both channels use the same late writer. `memory_layers` controls
attention readers; `recurrent_layers` controls preceding-token mergers.
At overlapping layers, attention runs before recurrence, both before the MLP.
See [the hybrid contract](MEMORY_ATTENTION.md#7-optional-recurrent-memory-hybrid).

There are no separately named hybrid models or active hybrid benchmark arms.

## Shared multipass causal invariant

Pass 1 is the current TinyMistral stream. Pass `k>1` consumes a completed
previous-pass source sequence with architecture-specific routing. This allows sequence-parallel training. Exact
cached K-pass inference snapshots lower-stream feedback before computing the
same physical position in higher streams, so no same-position lower-stream
state leaks upward. Live Feedback inference closes the final stream only
after the exact K-pass prefill boundary.
