# Architecture contracts

Experiment-specific learning rates, token budgets, pass depth, initialization,
and checkpoint ancestry belong under `benchmarks/`. This file defines only the
reusable architecture surface. The full bank/control-token contract is in
`BANK_MEMORY.md`.

The optional training-only latent objective is specified in
`NEXT_MEMORY_PREDICTION.md`.

## Vanilla

One ordinary TinyMistral causal pass with no architecture-added parameters.

## FBT

An independent multipass comparison based on asymmetric latent feedback. It is
not part of the bank family. Later-pass fused inputs are RMS-normalized before
entering the backbone, while position zero retains its ordinary token
embedding. FBT implements the same exact cached K-stream and collapsed
recurrent inference interfaces as the other one-state feedback variants.

## Retired one-state controls

MemoryAdd and BankAddHybrid are retained only as historical implementation
controls. They are not part of the active experiment pipeline or current
results; new comparisons use adaptive Recirculation and the
Recirculation–Bank hybrid instead.

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

## Bank

`BankVariant` has one shared identity-initialized bias-free writer

```text
m = W_write h
```

and one independent GQA bank reader at each configured `memory_layers` index.
`memory_layers: all` expands to every decoder layer; `[3, 7]` is the default in
the active experimental pipeline. Every selected reader consumes the same
previous-pass top-layer bank. Within a selected decoder layer the bank residual
is applied after the ordinary self-attention residual and before the MLP.

Reader output projections are zero-initialized, so every pass is exact vanilla
at construction. Q/K/V remain normally initialized and begin receiving
gradients after an output projection has moved away from zero.

Cross-attention uses sequence-anchored RoPE by default. Query rotations use the
current linguistic sequence position and key rotations use the original write
position; compact bank indices are never used as positions. In memory-token
mode a control slot inherits the preceding linguistic boundary so inserting
control computation does not inflate memory age. `memory_position_encoding:
none` is retained only as an explicit ablation.

The architecture has three write policies:

- `dense`: write every ordinary position;
- `periodic`: write positions satisfying `(t + 1) % C == 0`;
- `memory_token`: write only explicit input-only `<MEM>` positions.

`memory_window=W` counts committed bank records, not source-token distance.
Every bank read is strict-past: a record written at physical position `t` is
first available to position `t+1`.

Dense and periodic C=1 are the same implementation and are required to be
numerically identical with matching weights.

For full-sequence MPS training, dense writes use the direct strict-past local
window path because the bank is already one record per token. This avoids the
compact-bank gather used by sparse periodic writes. It makes dense Bank faster
than periodic-32 Bank on the development Mac, although the per-layer readers
still make it more expensive than the retired one-state control.

### BankAddHybrid (retired)

`BankAddHybridVariant` is the same bank plus the MemoryAdd path. There is no
gate, controller, or fusion MLP between the channels.

For ordinary-token sequences its fast path is ordinary MemoryAdd. With explicit
memory slots, `<MEM>` does not advance the fast state. For

```text
A <MEM> B
```

previous-stream `h_A` supplies the Add residual to both `<MEM>` and B;
`h_MEM` writes the slow bank; B then becomes the next fast state. This preserves
a clean distinction between ordinary-token fast recurrence and explicit bank
writes.

## Shared multipass causal invariant

Pass 1 is the current TinyMistral stream. Pass `k>1` consumes a completed
previous-pass top-state sequence, allowing sequence-parallel training. Exact
cached K-pass inference snapshots lower-stream feedback before computing the
same physical position in higher streams, so no same-position lower-stream
state leaks upward. Collapsed recurrent inference closes the final stream only
after the exact K-pass prefill boundary.
