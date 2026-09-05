# Architecture contracts

This map separates active model families from supported controls and legacy implementations.
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

## FBT (retired from active studies)

An independent multipass comparison based on asymmetric latent feedback. It is
not part of the Memory Attention family. Later-pass fused inputs are RMS-normalized before
entering the backbone, while position zero retains its ordinary token
embedding. FBT implements the same exact cached K-stream and collapsed
Live Feedback inference interfaces as the other one-state feedback variants.

## Retired one-state controls

Standalone MemoryAdd remains for historical checkpoint compatibility.
It is not part of new studies. The named embedding-add and middle-layer
attention hybrids have been deleted; their checkpoints need the original revision.

### MemoryAdd

For pass `k > 1`, current token embedding `e_t` receives the immediately
preceding previous-pass top state:

```text
x_t = e_t + W_A RMSNorm(h^(k-1)_(t-1))
```

`W_A` is bias-free and zero-initialized. Position zero receives a zero recurrent
residual.

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
self-attention and before the MLP at `memory_layers`. The active study uses
`memory_layers: [3]` and `[3, 7]` separately with `memory_window: 1`. See
[RECURRENT_MEMORY.md](RECURRENT_MEMORY.md) for initialization, gradient flow,
comparison limits, and inference semantics.

## Legacy middle-layer recirculation

This shifted multipass implementation remains as legacy source for interpreting
historical records. It is no longer an active config or model-factory choice,
and its same-token paper replay/BPTT implementation has been deleted. New
recurrent comparisons use the late-memory contract above.

`RecirculationVariant` captures the output of a configured source decoder
layer and, on later passes, right-shifts it by one position before mixing it
into an earlier destination layer. The source is L2 norm-matched to the
destination state and combined with fixed coefficient `alpha`:

```text
h_destination <- alpha * norm_match(h_source[t-1])
                  + (1 - alpha) * h_destination
```

Fixed mode has no added trainable parameters. It is a Phase-B control with
`0 <= destination_layer < source_layer` and `alpha` in `[0, 1]`.

The configurable `recirculation_mode: adaptive` setting implements the
conditional vector alpha/beta variant from Mozer et al. A two-hidden-layer
GELU MLP with input LayerNorm consumes the concatenated source and destination
states and predicts independent sigmoid-bounded coefficient vectors:

```text
(alpha_t, beta_t) = sigmoid(MLP([source_t, destination_t]))
h_destination <- alpha_t * norm_match(source_t)
                  + beta_t * destination_t
```

Its output head is initialized so adaptive mode starts at the fixed mixture
(`alpha=recirculation_alpha`, `beta=1-alpha`). The controller is an added
parameter group: Phase A freezes the TinyMistral backbone.
Phase B fine-tunes the full model. Fixed mode remains the default.

Historical controller-only configs used:

```yaml
variant: recirculation
phase: A
recirculation_mode: adaptive
recirculation_source_layer: 6
recirculation_destination_layer: 3
```

## Memory Attention

`dense_memory_attention`, `strided_memory_attention` and
`dense_and_strided_memory_attention` are presets of `memory_attention`, using
one model implementation. They share a late writer and selected
post-self-attention readers. Memory-token attention is also supported, but is
not an active frozen arm. See [Memory Attention](MEMORY_ATTENTION.md) for
masks, retention, initialization, gradient flow, aliases and input-only MEM semantics.

### Optional recurrent-memory hybrid

Set `recurrent_merger` and explicit `recurrent_layers` on a Memory Attention
configuration. Both channels use the same late writer. `memory_layers` controls
attention readers; `recurrent_layers` controls preceding-token mergers.
At overlapping layers, attention runs before recurrence, both before the MLP.
MEM writes attention memory but does not advance the recurrent record.
See [the hybrid contract](MEMORY_ATTENTION.md#10-optional-recurrent-memory-hybrid).

There are no separately named hybrid models or active hybrid benchmark arms.

## Shared multipass causal invariant

Pass 1 is the current TinyMistral stream. Pass `k>1` consumes a completed
previous-pass source sequence with architecture-specific routing. This allows sequence-parallel training. Exact
cached K-pass inference snapshots lower-stream feedback before computing the
same physical position in higher streams, so no same-position lower-stream
state leaks upward. Live Feedback inference closes the final stream only
after the exact K-pass prefill boundary.
