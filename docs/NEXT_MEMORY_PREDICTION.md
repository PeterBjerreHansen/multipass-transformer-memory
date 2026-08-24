# Next-memory prediction

Next-memory prediction (NMP) is an optional training-only auxiliary objective.
It predicts the model's own future memory features while next-token prediction
(NTP) remains active. It does not change `forward`, generation, cached
recurrence, or the historical bank-state format.

## Causal objective

The prediction head receives only the current pass's final hidden state:

```text
prediction at t = P(h_t)
```

It never receives the next token, a next-token embedding, or a feature computed
from a future position. Targets are always detached. The default
`shared_final` mode uses the final computed pass as the teacher for every pass.

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

Memory-attention NMP predicts the first strictly later stored memory record:

```text
w(t) = first write position greater than physical position t
P_memory(h_t^k) -> stop_gradient(writer(s_w(t)^K))
```

For `bank` (or its public alias `memory_attention`), the configured dense,
periodic, or explicit-memory-token write policy defines `w(t)`. Multiscale
Memory Attention uses its dense source stream. The target is post-writer
because that is the representation read by cross-pass attention. Memory-attention
targets keep their post-writer scale. Query positions must be linguistic, but
write positions may be controls.
In memory-token mode, a future `<MEM>` can have linguistic distance zero from
the preceding token even though its physical index is strictly greater.

Memory-attention loss is target-balanced. Guesses for one write are averaged first, actual
write events are then averaged within each example, and valid examples are
averaged last. This prevents longer spacing from increasing an event's loss
mass merely because it has more guesses.
Headline target/error diagnostics use the same event-within-example measure.
Explicit `event_*` and `query_*` metrics expose both measures; distance-bin
diagnostics are likewise labeled.

Both objectives use Smooth-L1 over latent dimensions. NMP pass weights are
independent of NTP pass weights and are uniform when no objective-specific map
is configured. Configure them by sampled pass count when needed:

```yaml
recurrent_nmp_pass_loss_weights_by_k:
  2: [0.5, 0.5]
bank_nmp_pass_loss_weights_by_k:
  2: [0.5, 0.5]
```

`nmp_target_mode: shared_final` preserves the formulation above. The
`same_pass` ablation instead pairs each predictor with the future memory from
its own pass:

```text
P(h_t^k) -> stop_gradient(memory_(future)^k)
```

Pass weights only change how pass-specific losses are combined. A map such as
`2: [0.0, 1.0]` implements final-pass-only NMP without another target mode.
Every by-K vector must contain exactly K entries.

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
nmp_target_mode: shared_final
nmp_detach_predictor_input: false
nmp_eval_passes: [2]
```

The ramp multiplies both configured NMP weights linearly from zero to one. NTP
continues throughout the run. Start with one small auxiliary coefficient and
confirm that `*_nmp_weighted_loss` remains subordinate to `ntp_loss` before
trying a dual-head hybrid.

Supported objectives are:

| Variant | Recurrent NMP | Memory-attention NMP |
| --- | ---: | ---: |
| `recirculation` | yes | no |
| `memory_attention` (`bank`) | no | yes |
| `memory_attention_multiscale` (`bank_multiscale`) | no | yes |
| `memory_attention_recirculation_hybrid` (`bank_recirculation_hybrid`) | yes | yes |
| `vanilla`, `sparse_swa`, `fbt` | no | no |

Each enabled objective gets an independent
`RMSNorm -> Linear -> GELU -> Linear -> GELU -> Linear` head. Heads have the
same architecture but do not share parameters. The final projection is
zero-initialized. Head construction uses an isolated seed derived from
`architecture_seed`.

## Gradients and checkpoints

Targets are stop-gradient. Predictor inputs normally remain connected, so NMP
gradients can shape `h_t` and any causal memory pathway that helped create it.
Set `nmp_detach_predictor_input: true` for the head-only placebo. For Memory Attention,
the writer target branch is detached, while the same writer remains reachable
through earlier written memories that contributed to later hidden states.

NMP heads are architecture-added parameters and train in both phases. Phase A
freezes the pretrained backbone; Phase B trains all parameters.

With both weights at zero, no heads are instantiated and historical state keys
remain unchanged. Exact resume is always strict. For `init_from` only, a whole
new head may be absent from an older checkpoint and retains its deterministic
fresh initialization. Partial heads and every other missing or unexpected key
remain errors. `init_from` also compares semantic architecture fields,
including write policy, memory layers/windows, and Recirculation routing, after
canonicalizing historical variant names. A source that already contains NMP is
rejected unless `allow_nmp_warm_start: true` is recorded explicitly.
Initialization provenance records the compatibility view and every freshly
initialized key.

When `nmp_eval_passes` is configured, validation evaluates the online detached
target on held-out blocks at each declared fixed K. These measurements report
generalization to unseen sequences, but the teacher still moves with the model;
interpret them alongside target-drift diagnostics and NTP NLL.

## Leakage checks

The test suite asserts that:

- changing tokens after `t` cannot change any recurrent or memory-attention
  prediction at or before `t`, across K=2/3 adaptive Recirculation,
  dense/periodic/memory-token/multiscale Memory Attention, and hybrid head modes;
- recurrent controls are skipped and memory-attention targets are physically strict-future;
- shared-final and same-pass modes select the declared teacher exactly;
- Recirculation uses its internal captured source, not the top hidden state;
- Memory Attention targets the post-writer representation;
- sparse batches with no future write produce a zero auxiliary loss and remain
  numerically valid;
- target tensors receive no gradient; the detached-input placebo removes only
  the auxiliary backbone gradient while preserving head gradients;
- enabled heads do not change ordinary forward logits.
