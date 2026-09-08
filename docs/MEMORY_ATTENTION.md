# Memory Attention

This is the architecture contract for cross-pass attention over strict-past memory records.
The active models use the following names:

| Public variant | Access pattern |
| --- | --- |
| `dense_memory_attention` | Recent previous-pass states |
| `strided_memory_attention` | Regularly spaced previous-pass states |
| `dense_and_strided_memory_attention` | Recent dense states plus older strided states |
| `memory_attention` | Explicit configuration of any supported pattern below |

All attention names are presets of the same `memory_attention` implementation:
`MemoryAttentionVariant` in
[`memory_attention.py`](../src/tiny_mistral_mptt/variants/memory_attention.py).
There is no separate dense-and-strided model class or factory branch.
The public name stays in experiment metadata; the resolved settings determine behavior.

`memory_pattern: dense|strided|dense_and_strided` selects ordinary-record access.
The descriptive names supply their
pattern and write mode. Contradictory explicit settings fail instead of overriding
the name. `memory_attention` accepts explicit settings without a preset.

Retired serialized names are translated at the input boundary by
[`compatibility.py`](../src/tiny_mistral_mptt/compatibility.py).
They are not runtime implementation keys, Python class aliases or advertised variants.
New config metadata uses current names; historical artifacts keep their recorded identities.
Old Python import paths and state-field aliases have been removed. Pure-attention tensor parameter
keys are unchanged, so serialized weights remain loadable when the architecture matches.
Checkpoint comparison resolves aliases but retains pattern, reader layout and optional
recurrent settings as architecture fields. The deleted named hybrids are not aliases:
their checkpoints require the original repository revision.

See [the architecture map](ARCHITECTURES.md) for controls and preserved reference implementations.
Study-specific reader locations and capacities belong in the [frozen protocol](../benchmarks/development/frozen_backbone_comparison/README.md).

## 1. Shared Memory Attention architecture

All write policies use the same components:

- one shared, identity-initialized, bias-free `MemoryAttentionWriter`.
- one independent `MemoryAttentionReader` at each configured `memory_layers` index.
- one chronological memory state of previous-pass writer outputs during full-sequence
  execution.
- a bounded chronological `MemoryAttentionState` during cached/recurrent execution.

Each selected reader layer performs ordinary self-attention first, adds its residual, reads the
memory records, fuses the memory read with the residual stream, then performs the normalized MLP
residual. The reader is Mistral-shaped GQA and has its own query/memory RMSNorm plus Q/K/V/O
projections.

During cached inference, each retained memory record is projected once per
reader when it is created or appended. RoPE rotates the key with the record's
original linguistic sequence position before it enters the cache. Subsequent
cached reads project and rotate only the query. The raw-memory and projected
cache paths are required to be numerically identical.

The default `memory_reader_initialization: zero_output` uses random Q/K/V and a
zero output projection. Memory Attention then starts as an exact no-op retrofit:
pass 2 and deeper passes equal the SWA Transformer at construction. The output
projections can learn on the first optimizer step. Gradients can then reach
Q/K/V and the writer as those projections become nonzero.

The explicit `memory_reader_initialization: aligned_gqa` alternative is nonzero.
Within each GQA group, Q/K/V start in the same pooled backbone coordinates and
O maps the repeated query heads back with reciprocal group scaling. It is an
orthogonal projection onto the group-shared subspace, not a literal identity
when the reader has fewer KV heads than query heads.

Let `a` be the raw attention output and `d` the destination residual stream.
The four supported fusion modes are:

```text
residual:          d' = d + a
destination_gated: d' = beta(a,d) * d + a
attention_gated:   d' = d + alpha(a,d) * a
dual_gated:        d' = beta(a,d) * d + alpha(a,d) * a
```

The token- and feature-wise sigmoid gates are produced by the same controller
form over `[a,d]`: LayerNorm followed by two GELU MLP layers and a gate-output
projection. `alpha` initializes to 0.1 and `beta` to 0.9 using zero output
weights and biased logits. The gates are independent and are not constrained
to sum to one. Single-gate modes emit only their named gate; `dual_gated` emits
both. Gated modes require `memory_attention_controller_hidden_size`, while
residual fusion has no controller.

Attention fusion does not norm-match `a` to `d`. This keeps all four modes on
the same raw reader output and makes the study a direct decomposition of which
contribution is gated. Positions with no strictly-past record bypass every
fusion exactly. Reader initialization and fusion are separate architecture
axes and are checked when loading weights. With the default zero-output reader,
residual and attention-gated fusion are exact no-ops at construction; a
destination gate can still alter available positions through its beta path.

A memory record is

```text
m_s = W_write h_s
```

where `h_s` is the final normalized backbone state from the source stream. For dense and strided configurations,
`memory_window=W` is the maximum number of committed records presented to a
query. It is not a token-distance window. Dense-and-strided Memory Attention has capacity `D+S`.

## 2. Reader placement and memory positions

```yaml
memory_layers: [3, 7]             # or: all
memory_position_encoding: rope    # default; explicit ablation: none
```

Layer indices are zero-based and unique. A non-reader decoder layer performs
ordinary self-attention and MLP computation without allocating memory-attention
reader parameters or projected memory K/V.

Memory RoPE is anchored to the original linguistic sequence, never to compact
memory-record order. A record written for linguistic position 511 remains position 511
even if it is the third retained record. `MemoryAttentionBatch` and cached
`MemoryAttentionState` carry these coordinates through compaction, bounded eviction, and
incremental decoding.

## 3. Write policies

These are configurations of `memory_attention`, not separate implementations.
Dense-and-strided uses a dense source stream and the retention policy in section 3.3.
Use the descriptive names below, or `variant: memory_attention` with explicit settings.

### 3.1 Dense

```yaml
variant: dense_memory_attention
# Equivalent: variant: memory_attention, memory_pattern: dense
memory_write_mode: dense
```

Every ordinary physical position writes. This is also the C=1 endpoint of the
strided policy.

### 3.2 Strided

```yaml
variant: strided_memory_attention
# Equivalent: variant: memory_attention, memory_pattern: strided
memory_write_mode: strided
memory_write_stride: 8
```

For zero-based physical position `t`, a write occurs when
`(t + 1) % C == 0`. With no control positions this means C=8 writes at
7, 15, 23, ... .

### 3.3 Dense-and-strided retention

```yaml
variant: dense_and_strided_memory_attention
# Equivalent: variant: memory_attention, memory_pattern: dense_and_strided
memory_dense_window: 32
memory_sparse_stride: 32
memory_sparse_window: 32
memory_layers: [3, 7]
memory_position_encoding: rope
```

Every previous-pass top state is written through the shared memory writer.
Query `t` reads dense positions `[t-D,t)` plus the last `S` positions strictly
older than `t-D` for which `(s+1) % C == 0`. The regions are concatenated in
chronological order and processed by one memory-attention reader and one softmax. The
sparse region is a retention policy over the dense source stream, not a second
writer or reader.

`memory_dense_window + memory_sparse_window` is the cached Memory Attention capacity.
During decode, an aging dense record survives only when it meets the strided
policy and remains among the last `S` sparse records. Raw memory and per-reader
projected K/V stay aligned with their original linguistic positions.

## 4. Strict read-compute-write timing

Memory Attention causality is always:

```text
READ old memory -> COMPUTE current hidden -> optionally WRITE current hidden
```

A current position never reads its own newly written record. In full-sequence
multipass execution this is represented by `writes_before[b,t]`, the number of
records committed strictly before physical position t. In cached execution the
old bounded `MemoryAttentionState` is passed to the reader and the append happens only
after the token hidden is complete.

## 5. Full-sequence versus Live Feedback execution

During training and exact K-pass evaluation, pass k reads memory-attention/recurrent feedback
constructed from completed pass k-1. The same-position source state is never
visible. This preserves parallel sequence training.

Cached exact K-pass inference keeps K self-attention streams and updates
feedback only after all streams process the current physical position. It is
tested against full-prefix recomputation.

Live Feedback decoding starts from the exact K-pass prompt prefill. For K>1 its first
continuation transition therefore matches exact K-pass inference; after that,
the final live stream feeds its own newly produced states into the feedback
machinery. K=1 feedback is also supported and uses a real prompt memory. Prompt
K is independent of the `standard` versus `feedback` continuation mode.

## 6. Data and compute accounting

Packed data contains ordinary vocabulary IDs only. Each input token occupies one
model position. Token budgets count presentations, including repeated corpus
passes; token-equivalent compute also counts the number of refinement passes.
See [training accounting](TRAINING.md#token-accounting).

## 7. Optional recurrent-memory hybrid

Any Memory Attention configuration can also enable a preceding-token memory merger:

```yaml
variant: dense_memory_attention
memory_window: 64
memory_layers: [3, 7]             # attention readers
recurrent_merger: projected_residual  # or recirculation
recurrent_layers: [3]             # required, explicit recurrent readers
```

Leave both recurrent fields unset for pure Memory Attention. This is an optional
configuration, not another public variant or an additional frozen benchmark arm.
The implementation is
[`MemoryAttentionRecurrentHybridVariant`](../src/tiny_mistral_mptt/variants/memory_attention_recurrent_hybrid.py).

Both channels use the same final normalized top state and the same writer weights.
Attention selects strict-past records according to its configured pattern.
The recurrent channel selects only the preceding ordinary token's emitted record,
using the same merger modules as standalone [recurrent memory](RECURRENT_MEMORY.md).
It does not capture a middle-layer source or add memory to token embeddings.

At a layer shared by both channels the order is:
self-attention residual, memory-attention residual, recurrent merger, MLP residual.
Separate reader layers are allowed. With zero attention output projections and
ordinary-token input, the hybrid equals standalone recurrent memory with matching
writer, merger and recurrent layers.

During full-sequence training the shared writer is applied separately to the
attention-selected source and the recurrent source. This keeps writer gradients
enabled when Phase A detaches the first backbone pass. Cached recurrence stores
an already-emitted record; it must not apply the writer again when reading.

Channel-specific diagnostics can zero or mismatch either source independently.
The deleted embedding-add and middle-layer hybrids have different computation;
their names and checkpoints are rejected, not redirected here.

## 8. Phase A gradients

In dense/strided Phase A, pass 1 contains no architecture-added parameter, so
it can run under `no_grad()` while the frozen backbone supplies the source state.

## 9. Validation

The required causality, endpoint-equivalence, masking, gradient, cache, and
resume checks are listed in [VALIDATION.md](VALIDATION.md). Run `make check`
before interpreting quality results.

Explicit memory-token attention is preserved only as
[reference source](../historical/implementations/feedback/README.md). Its inserted
positions, special labels and decoding behavior are absent from this runtime.
