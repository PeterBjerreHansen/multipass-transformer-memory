# Next-memory prediction

Next-memory prediction (NMP) is an optional training-only auxiliary objective.
It predicts the model's own future memory features while next-token prediction
(NTP) remains active. It does not change `forward`, generation, cached
recurrence, or the bank-state format.

## Causal objective

The prediction head receives only the current pass's final hidden state:

```text
prediction at t = P(h_t)
```

It never receives the next token, a next-token embedding, or a feature computed
from a future position. For every pass, the target comes from the final computed
pass and is detached.

Recurrent NMP predicts the memory emitted at the next linguistic token:

```text
P_recurrent(h_t^k) -> stop_gradient(RMS(r_(t+1)^K))
```

`r` is semantic and architecture-specific. Adaptive Recirculation uses its
captured source-layer state; the active hybrid uses its recurrent component.
The retired MemoryAdd and BankAddHybrid controls use their historical recurrent
states only for checkpoint compatibility. The default target normalization uses
the same parameter-free variance calculation as Mistral RMSNorm, without a
learned gain. This removes the residual-stream amplitude that `_norm_match`
discards at routing time. Set `recurrent_nmp_target_normalization: none` for the
raw-state ablation. Explicit `<MEM>` controls are skipped when finding the next
linguistic token.

Bank NMP predicts the first strictly later stored memory:

```text
w(t) = first write position greater than physical position t
P_bank(h_t^k) -> stop_gradient(writer(s_w(t)^K))
```

For `bank`, the configured dense, periodic, or explicit-memory-token write
policy defines `w(t)`. Multiscale Bank uses its dense source stream. The target
is post-writer because that is the representation read from the Bank. Bank
targets keep their post-writer scale. Query positions must be linguistic, but
write positions may be controls.
In memory-token mode, a future `<MEM>` can have linguistic distance zero from
the preceding token even though its physical index is strictly greater.

Bank loss is target-balanced. Guesses for one write are averaged first, actual
write events are then averaged within each example, and valid examples are
averaged last. This prevents longer spacing from increasing an event's loss
mass merely because it has more guesses.

Both objectives use Smooth-L1 over latent dimensions. NMP pass weights are
independent of NTP pass weights and are uniform when no objective-specific map
is configured. Configure them by sampled pass count when needed:

```yaml
recurrent_nmp_pass_loss_weights_by_k:
  2: [0.5, 0.5]
bank_nmp_pass_loss_weights_by_k:
  2: [0.5, 0.5]
```

Every pass still predicts the same detached final-pass target; weighting only
changes how the pass-specific prediction losses are combined. Per-pass metrics
expose the individual losses and weights.

## Configuration

NMP must start as a continuation of an NTP-trained memory model. An enabled
configuration therefore requires `init_from` or `resume_from` and a positive
token-based loss ramp:

```yaml
init_from: benchmarks/.../checkpoint.pt
recurrent_nmp_weight: 0.05
bank_nmp_weight: 0.0
nmp_projection_factor: 1.3
nmp_warmup_tokens: 262144
```

The ramp multiplies both configured NMP weights linearly from zero to one. NTP
continues throughout the run. Start with one small auxiliary coefficient and
confirm that `*_nmp_weighted_loss` remains subordinate to `ntp_loss` before
trying a dual-head hybrid.

Supported objectives are:

| Variant | Recurrent NMP | Bank NMP |
| --- | ---: | ---: |
| `recirculation` | yes | no |
| `bank` | no | yes |
| `bank_multiscale` | no | yes |
| `bank_recirculation_hybrid` | yes | yes |
| `vanilla`, `sparse_swa`, `fbt` | no | no |

Each enabled objective gets an independent
`RMSNorm -> Linear -> GELU -> Linear -> GELU -> Linear` head. Heads have the
same architecture but do not share parameters. The final projection is
zero-initialized. Head construction uses an isolated seed derived from
`architecture_seed`.

## Gradients and checkpoints

Targets are stop-gradient. Predictor inputs are not detached, so NMP gradients
can shape `h_t` and any causal memory pathway that helped create it. For Bank,
the writer target branch is detached, while the same writer remains reachable
through earlier written memories that contributed to later hidden states.

NMP heads are architecture-added parameters and train in both phases. Phase A
freezes the pretrained backbone; Phase B trains all parameters.

With both weights at zero, no heads are instantiated and historical state keys
remain unchanged. Exact resume is always strict. For `init_from` only, a whole
new head may be absent from an older checkpoint and retains its deterministic
fresh initialization. Partial heads and every other missing or unexpected key
remain errors. The initialization provenance records every freshly initialized
key.

## Leakage checks

The test suite asserts that:

- changing tokens after `t` cannot change any recurrent or bank prediction at
  or before `t`, across multiple pass depths;
- recurrent controls are skipped and bank targets are physically strict-future;
- all passes use one shared detached final-pass target;
- Recirculation uses its internal captured source, not the top hidden state;
- Bank targets the post-writer representation;
- sparse batches with no future write produce a zero auxiliary loss and remain
  numerically valid;
- target tensors receive no gradient, while predictor inputs and the causal
  writer-to-reader path do;
- enabled heads do not change ordinary forward logits.
