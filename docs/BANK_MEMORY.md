# Bank memory and explicit `<MEM>` slots

This is the authoritative contract for the active `bank` and
`bank_multiscale` variants. See [ARCHITECTURES.md](ARCHITECTURES.md) for the
retired `bank_add_hybrid` interface.

## 1. Shared bank architecture

All write policies use the same components:

- one shared, identity-initialized, bias-free `BankWriter`.
- one independent `BankReader` at each configured `memory_layers` index.
- one chronological bank of previous-pass writer outputs during full-sequence
  execution.
- a bounded chronological `BankState` during cached/recurrent execution.

Every layer performs ordinary self-attention first, adds its residual, reads the
bank, adds the bank residual, then performs the normalized MLP residual. The
reader is Mistral-shaped GQA and has its own query/memory RMSNorm plus Q/K/V/O
projections.

During cached inference, each retained bank record is projected once per
reader when it is created or appended. RoPE rotates the key with the record's
original linguistic sequence position before it enters the cache. Subsequent
cached reads project and rotate only the query. The raw-memory and projected
cache paths are required to be numerically identical.

Reader output projections are zero-initialized. Bank starts as an
exact no-op retrofit: pass 2 and deeper passes equal vanilla at construction.
The output projections learn on the first optimizer step; Q/K/V and writer
gradients become active after those projections move away from zero.

A bank record is

```text
m_s = W_write h_s
```

where `h_s` is a top-layer source-stream state. For `variant: bank`,
`memory_window=W` is the maximum number of committed records presented to a
query. It is not a token-distance window. Multiscale Bank has capacity `D+S`.

## 2. Reader placement and memory positions

```yaml
memory_layers: [3, 7]             # or: all
memory_position_encoding: rope    # default; explicit ablation: none
```

Layer indices are zero-based and unique. A non-reader decoder layer performs
ordinary self-attention and MLP computation without allocating Bank-reader
parameters or projected Bank K/V.

Memory RoPE is anchored to the original linguistic sequence, never to compact
bank order. A record written for linguistic position 511 remains position 511
even if it is the third retained bank record. In memory-token mode a `<MEM>`
slot inherits the preceding linguistic boundary. `BankBatch` and cached
`BankState` carry these coordinates through compaction, bounded eviction, and
incremental decoding.

## 3. Write policies

The three policies below apply to `variant: bank`. `bank_multiscale` instead
uses a dense source stream and the retention policy in section 3.4.

### 3.1 Dense

```yaml
memory_write_mode: dense
```

Every ordinary physical position writes. This is also the C=1 endpoint of the
periodic policy.

### 3.2 Periodic

```yaml
memory_write_mode: periodic
memory_write_stride: 8
```

For zero-based physical position `t`, a write occurs when
`(t + 1) % C == 0`. With no control positions this means C=8 writes at
7, 15, 23, ... .

### 3.3 Explicit memory token

```yaml
memory_write_mode: memory_token
memory_write_stride: 8
memory_token_visibility: visible   # or write_only
```

The data view inserts one `<MEM>` after each complete group of C linguistic
tokens when another linguistic token remains in that block. Only MEM positions
write the bank.

### 3.4 Multiscale dense-recent/sparse-old retention

```yaml
variant: bank_multiscale
memory_dense_window: 32
memory_sparse_stride: 32
memory_sparse_window: 32
memory_layers: [4, 7]
memory_position_encoding: rope
```

Every previous-pass top state is written through the shared `BankWriter`.
Query `t` reads dense positions `[t-D,t)` plus the last `S` positions strictly
older than `t-D` for which `(s+1) % C == 0`. The regions are concatenated in
chronological order and processed by one Bank reader and one softmax. The
sparse region is a retention policy over the dense source stream, not a second
writer or reader.

`memory_dense_window + memory_sparse_window` is the cached Bank capacity.
During decode, an aging dense record survives only when it meets the periodic
policy and remains among the last `S` sparse records. Raw memory and per-reader
projected K/V stay aligned with their original linguistic positions.

## 4. `<MEM>` is input-only

Let the pretrained vocabulary size be `V`.

```text
ordinary input IDs: 0 ... V-1
<MEM> input ID:     V
LM output classes:  0 ... V-1
```

The pretrained embedding table and LM head are not resized. Bank variants in
memory-token mode own one architecture-added learned `memory_token_embedding`;
ID V selects that vector. The embedding is currently initialized to zero and
learns as an added parameter.

Because the LM head remains size V, `<MEM>` cannot receive probability mass or
be sampled as a language token.

## 5. Language loss skips control slots

The physical transformer sequence and linguistic prediction sequence are not
the same. For

```text
physical positions:  A    <MEM>    B    C
LM labels:            B    IGNORE   C    IGNORE
```

A predicts the next **linguistic** token B across the control slot. The MEM
hidden state has no direct LM prediction objective. The final ordinary position
also has no target inside the packed block.

More generally, only ordinary positions predict, and each predicts the nearest
ordinary token strictly to its right. Every MEM position receives `ignore_index`
in cross-entropy. This is implemented as position-aligned labels rather than an
ordinary one-position shift.

The MEM representation can still receive gradients indirectly. In visible mode
future ordinary-token losses can flow through self-attention into MEM; in both
modes later recurrent/bank-mediated losses can flow through the bank reader,
writer, and MEM state.

## 6. Self-attention visibility

### `visible`

`<MEM>` is an ordinary causal self-attention K/V position. Later tokens may use
its hidden state locally as well as through the persistent bank. Thus any gain
can include both dedicated latent compute and improved bank storage.

### `write_only`

`<MEM>` remains a transformer query and can read preceding causal context, but
its self-attention K/V is marked invalid. No query uses MEM as an ordinary
self-attention key/value; the MEM input/residual path still exists and its
hidden state still writes the bank.

This isolates the persistent memory route more cleanly. The boolean key-validity
mask is supported by the reference, local O(TW), and FlexAttention full-sequence
backends. Cached KV entries retain their physical/RoPE position and carry the
same validity bit, so masking does not collapse sequence positions.

## 7. Strict read-compute-write timing

Bank causality is always:

```text
READ old bank -> COMPUTE current hidden -> optionally WRITE current hidden
```

A current position never reads its own newly written record. In full-sequence
multipass execution this is represented by `writes_before[b,t]`, the number of
records committed strictly before physical position t. In cached execution the
old bounded `BankState` is passed to the reader and the append happens only
after the token hidden is complete.

For memory-token input

```text
A <MEM> B
```

`h_MEM` may write a bank record, and B is the first physical position that can
read that record.

## 8. Full-sequence versus recurrent execution

During training and exact K-pass evaluation, pass k reads bank/fast feedback
constructed from completed pass k-1. The same-position source state is never
visible. This preserves parallel sequence training.

Exact incremental K-pass inference keeps K self-attention streams and updates
feedback only after all streams process the current physical position. It is
tested against full-prefix recomputation.

Collapsed recurrent inference starts from the exact K-pass prefill. Its first
continuation transition therefore matches exact K-pass inference; after that,
the final live stream feeds its own newly produced states into the recurrent
feedback machinery.

If a cached decode step consumes `<MEM>`, `next_token_logits` remain the logits
from the preceding ordinary position because MEM itself predicts nothing.

## 9. Data view and compute accounting

The stored Dolmino artifacts contain only ordinary linguistic IDs. A deterministic
`MemoryTokenPackedDataset` view inserts ID V at load time, preserving the
underlying linguistic token order and artifact provenance.

The control positions are **additional physical transformer positions**. For a
backing block with N linguistic tokens and cadence C:

```text
physical positions = N + floor((N - 1) / C)
```

For example, a 2048-linguistic-token block at C=8 becomes 2303 physical model
positions. This deliberately keeps the linguistic data dose fixed and makes the
extra MEM compute explicit; periodic and MEM-token experiments are not
compute-identical.

Training telemetry therefore separates:

```text
unique_tokens_seen       linguistic/data tokens
model_positions_seen     physical positions including MEM
token_equivalent_compute physical positions x effective passes
```

Run budgets and LR schedules use linguistic tokens. Throughput should report
both linguistic tokens/s and model positions/s.

## 10. Phase A wrinkle

In dense/periodic Phase A, pass 1 contains no architecture-added parameter, so
it can run under `no_grad()` while the frozen backbone supplies the source state.

In memory-token Phase A, the architecture-added MEM embedding participates in
pass 1. Pass-1 autograd must therefore remain enabled even though pretrained
backbone parameters stay frozen. With zero-initialized reader outputs, the MEM
embedding and writer have zero gradient on the first update and receive
nonzero bank-mediated gradients after the reader output path activates.

## 11. Validation

The required causality, endpoint-equivalence, masking, gradient, cache, and
resume checks are listed in [VALIDATION.md](VALIDATION.md). Run `make check`
before interpreting quality results.
