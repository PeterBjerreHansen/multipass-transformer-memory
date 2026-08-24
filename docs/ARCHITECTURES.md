# Architecture contracts

This file defines the reusable model interfaces. Study-specific settings belong
under `benchmarks/`. See [MEMORY_ATTENTION.md](MEMORY_ATTENTION.md) for the
public terminology and [BANK_MEMORY.md](BANK_MEMORY.md) for the compatibility
contract. [NEXT_MEMORY_PREDICTION.md](NEXT_MEMORY_PREDICTION.md) documents the
optional training-only objective.

## Vanilla

One ordinary TinyMistral causal pass with no architecture-added parameters.

## Sparse SWA control

`SparseSWAVariant` is a one-pass, parameter-free Transformer control. At the
selected `sparse_attention_layers`, the existing Mistral self-attention uses
one softmax over the union of its ordinary SWA keys and a bounded set of older
fixed-periodic keys. It reuses the pretrained Q/K/V/O projections and does not
add a Memory Attention reader or any cross-attention parameters.

For query `t`, SWA width `W`, sparse stride `C`, and sparse count `S`, the added
keys are the last `S` positions satisfying `(s + 1) % C == 0` and `s < t-W+1`.
The sparse region never duplicates a key in the local region. Cached decoding
retains `W-1` recent K/V entries plus at most `S` older periodic entries with
their absolute RoPE positions.

```yaml
variant: sparse_swa
sparse_attention_stride: 32
sparse_attention_window: 32
sparse_attention_layers: [3, 7]
```

## FBT

An independent multipass comparison based on asymmetric latent feedback. It is
not part of the Memory Attention family. Later-pass fused inputs are RMS-normalized before
entering the backbone, while position zero retains its ordinary token
embedding. FBT implements the same exact cached K-stream and collapsed
recurrent inference interfaces as the other one-state feedback variants.

## Retired one-state controls

MemoryAdd and BankAddHybrid remain for historical checkpoint compatibility.
They are not part of new studies.

### MemoryAdd

For pass `k > 1`, current token embedding `e_t` receives the immediately
preceding previous-pass top state:

```text
x_t = e_t + W_A RMSNorm(h^(k-1)_(t-1))
```

`W_A` is bias-free and zero-initialized. Position zero receives a zero recurrent
residual.

## Recirculation

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
parameter group: use Phase A to freeze the TinyMistral backbone, as in the
paper, or Phase B to fine-tune the full model. Fixed mode remains the default.

Paper-style controller-only adaptation can be configured as:

```yaml
variant: recirculation
phase: A
recirculation_mode: adaptive
recirculation_source_layer: 6
recirculation_destination_layer: 3
```

## Memory Attention

`BankVariant` (public alias `MemoryAttentionVariant`, `variant:
memory_attention`) has one shared identity-initialized bias-free writer

```text
m = W_write h
```

and one independent GQA memory-attention reader at each configured
`memory_layers` index.
`memory_layers: all` expands to every decoder layer. The original study used
`[3, 7]` for standalone Memory Attention models. Every selected reader consumes
the same previous-pass top-layer memory records. Within a selected decoder
layer the memory-attention residual is applied after the ordinary self-attention
residual and before the MLP.

Reader output projections are zero-initialized, so every pass is exact vanilla
at construction. Q/K/V remain normally initialized and begin receiving
gradients after an output projection has moved away from zero.

Cross-attention uses sequence-anchored RoPE by default. Query rotations use the
current linguistic sequence position and key rotations use the original write
position; compact memory-record indices are never used as positions. In memory-token
mode a control slot inherits the preceding linguistic boundary so inserting
control computation does not inflate memory age. `memory_position_encoding:
none` is retained only as an explicit ablation.

The architecture has three write policies:

- `dense`: write every ordinary position;
- `periodic`: write positions satisfying `(t + 1) % C == 0`;
- `memory_token`: write only explicit input-only `<MEM>` positions.

`memory_window=W` counts committed memory records, not source-token distance.
Every memory-attention read is strict-past: a record written at physical position `t` is
first available to position `t+1`.

Dense and periodic C=1 are the same implementation and are required to be
numerically identical with matching weights.

### Multiscale Memory Attention control

`MultiscaleBankVariant` (public alias `MultiscaleMemoryAttentionVariant`,
`variant: memory_attention_multiscale`) is the attention-only analogue of a
short/long-range recurrent–Memory Attention hybrid. It writes the same dense
previous-pass top-state stream as Dense Memory Attention, then presents each reader with a
non-overlapping union of the preceding `D` records and the last `S` older
fixed-periodic records. A single memory-attention reader and softmax cover both
regions.
The model does not add a second reader residual.

```yaml
variant: memory_attention_multiscale  # historical alias: bank_multiscale
memory_dense_window: 32
memory_sparse_stride: 32
memory_sparse_window: 32
memory_layers: [4, 7]
memory_position_encoding: rope
```

At query `t`, the dense region is `[t-D,t)`. Sparse positions satisfy
`s < t-D` and `(s+1) % C == 0`, with only the most recent `S` retained. The
maximum Memory Attention capacity is `D+S`. The approximate oldest direct reach is
`D + C*S` tokens. Vanilla SWA is unchanged.

### Memory Attention + MemoryAdd hybrid (retired)

`BankAddHybridVariant` (public alias `MemoryAttentionAddHybridVariant`) is the
same Memory Attention path plus the MemoryAdd path. There is no
gate, controller, or fusion MLP between the channels.

For ordinary-token sequences its fast path is ordinary MemoryAdd. With explicit
memory slots, `<MEM>` does not advance the fast state. For

```text
A <MEM> B
```

previous-stream `h_A` supplies the Add residual to both `<MEM>` and B;
`h_MEM` writes a slow memory record; B then becomes the next fast state. This preserves
a clean distinction between ordinary-token fast recurrence and explicit memory
writes.

## Shared multipass causal invariant

Pass 1 is the current TinyMistral stream. Pass `k>1` consumes a completed
previous-pass top-state sequence, allowing sequence-parallel training. Exact
cached K-pass inference snapshots lower-stream feedback before computing the
same physical position in higher streams, so no same-position lower-stream
state leaks upward. Collapsed recurrent inference closes the final stream only
after the exact K-pass prefill boundary.
