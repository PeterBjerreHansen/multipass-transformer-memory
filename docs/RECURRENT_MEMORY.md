# Late-emitted recurrent memory

This is the active recurrent architecture for the frozen comparison. It replaces
the middle-layer source used by the older `variant: recirculation` arm. The older
shifted multipass implementation remains for historical configs. Its paper-replay
training and inference policy has been deleted; neither defines this comparison.

## Shared memory contract

Both recurrent mergers and the attention variants use the same identity-initialized,
bias-free D-to-D writer on the final normalized backbone state:

```text
m_t^(k) = W_write h_final,t^(k)
```

At token `t` on pass `k`, a recurrent merger receives only `m_(t-1)^(k-1)`.
Position zero receives no feedback. There is no same-token mixing, accumulated
recurrent cache, or internal-layer source tap. Each arm trains its own writer;
the emission rule and initialization are shared, not the learned weights.

The active recurrent arms read at layer 3 (zero-based). Like Memory Attention,
the read occurs after self-attention and before the MLP. The legacy recirculation
arm inserted its mixture after the entire destination block, so the new baseline
changes that placement as well as its source. Historical results must not be
relabeled as results for the new baseline. Attention arms retain their existing
read layers `[3, 7]`; the overall comparison is not a pure access-pattern ablation.
The two recurrent arms, however, have identical routing and differ only in merger.

## Exactly two merger choices

`recurrent_merger: projected_residual` selects the new candidate:

```text
g = sigmoid(W_gate [RMSNorm(d), RMSNorm(m)] + b_gate)
d_new = d + g * W_project RMSNorm(m)
```

`W_project` starts at zero. The gate is normally initialized, with zero bias.
Every pass is therefore exactly vanilla at construction. The projection can
learn on the first update; gradients then reach the gate and writer. There is
no additional zero scalar gate that could block both learning paths.

`recurrent_merger: recirculation` selects the existing adaptive mixing rule:

```text
(alpha, beta) = sigmoid(MLP([m, d]))
d_new = alpha * norm_match(m, d) + beta * d
```

The controller uses the existing two-hidden-layer GELU MLP and input LayerNorm.
It starts at alpha=0.1 and beta=0.9; the coefficients are independent thereafter.
Unlike the projected residual, this merger perturbs later passes at initialization.
Report that initial condition rather than attributing all starting-loss differences
to learning. This comparison selects a useful merger design; it does not isolate
projection, residual preservation, gating, and initialization individually.

No additional merger candidates are in the active study. Revisit alternatives
only if the initial comparison provides a reason to do so.

## Configuration and training

```yaml
variant: recurrent_memory
recurrent_merger: projected_residual  # or recirculation
memory_window: 1
memory_layers: [3]
phase: A
training_forward: parallel_multipass
pass_schedule:
  - probabilities: {2: 0.9, 3: 0.1}
ntp_pass_loss_weights_by_k:
  2: [0.0, 1.0]
  3: [0.0, 0.0, 1.0]
validation_forward: parallel_multipass
eval_passes: 4
```

Phase A freezes the backbone and trains the writer and merger. The writer is
applied when the previous pass is consumed, outside the first-pass no-gradient
region. It therefore remains trainable even at K=2. A later unfrozen experiment
can use Phase B and `freeze_pretrained_until_tokens`, starting fresh from the
same pretrained checkpoint rather than continuing the frozen experiment.

Freezing restricts where adaptation can happen; it does not guarantee that
feedback carries useful contextual information. Check gate/projection values
and compare real feedback with zeroed or batch-mismatched records. Nonzero
weights alone are not evidence of useful memory.

The existing fixed-K cached inference and NLL paths are reused. Cached decoding
writes each new top-state record once and exposes it only to the next token.
No new inference mode or evaluator is introduced. Ordinary feedback decoding
remains supported with BOS-only or contextual prefill. Paper-recirculation
training and validation are removed from the codebase.

## Study and compatibility

The frozen comparison contains these two recurrent arms plus dense, strided and
multiscale Memory Attention. All use 2048-token blocks, identical training K
distributions and objectives, and K=4 headline NLL with K=1–4 diagnostics.
Each of the five arms gets the same four-value LR qualification budget.

The new recurrent configs have distinct arm IDs and output directories. Old
middle-layer weights cannot be resumed as late-memory weights. Checkpoint
compatibility includes the merger choice and read layers. Historical writer and
controller state-dict names remain unchanged by their shared-module extraction.

Scheduled snapshots and interruption-recovery checkpoints retain their existing,
separate policies. No training run was launched as part of this restructuring.

Cleanup 3–4 fixes K=1 state conversion and shares evaluation precision, scoring
and result identity. Snapshot publication is interruption-safe; see
[CLEANUP_STATUS.md](CLEANUP_STATUS.md). Initial-state and K=4
memory-use diagnostics are later additions. Both mergers already have a learned
writer projection; this comparison does not isolate projection versus no projection.
