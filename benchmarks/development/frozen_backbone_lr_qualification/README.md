# Frozen-backbone learning-rate qualification

This planned qualification selects the Phase-A learning rate for all five
feedback mechanisms before interpreting the frozen-backbone wiring comparison.
The backbone remains frozen throughout each run; only the added writer and
merger or Memory Attention parameters are optimized. The recurrent arms use
late emitted memory with projected-residual and adaptive-recirculation mergers.
This qualification uses the same
2,048-token `gpu_2048` artifact and 8x4 default optimizer batching as the
comparison study.

The attention arms all read at `[3, 7]`; the recurrent pair reads at `[3]`.
The combined attention arm uses the `dense_and_strided_memory_attention` preset
of `memory_attention`. These settings match the main comparison. Complete the bounded
real-trainer GPU preflight before this sweep; token-zero evaluation and expanded
memory diagnostics are not prerequisites. Feedback NLL remains disabled here;
the selected 5M/20M/100M feedback schedule applies only to the main 100M runs.

## Protocol

| Config field | Value |
| --- | --- |
| `phase` | `A` |
| `max_unique_tokens` | `5013504` |
| `batch_size` | `8` |
| `grad_accum_steps` | `4` |
| `eval_passes` | `4` |
| `eval_batches` | `64` |
| `eval_every_tokens` | `1048576` |
| `feedback_eval_at_tokens` | `null` |

Every arm consumes exactly 5,013,504 unique linguistic tokens. With 2,048-token
blocks and the nominal 8x4 policy this is 77 optimizer updates, with the final
update using the remaining partial accumulation. The pass schedule
is the shared wiring schedule: K=2 with probability 0.9 and K=3 with probability
0.1, with final-pass-only NTP loss weights. Validation uses deterministic K=4
whole-block pass-depth NLL on 64 held-out blocks; generation is not part of this
qualification.

The candidate grid is the same for every mechanism:

```text
3e-5, 1e-4, 3e-4, 1e-3
```

This is 20 independent fresh runs: five mechanisms times four constant
added-parameter learning rates, with one seed (`1337`) per setting. Each run
starts from the common pretrained checkpoint, not another qualification run.
The total tuning dose is 100,270,080 linguistic training tokens across the
20 runs; they reuse the same artifact, so this is not that many distinct corpus
tokens. The backbone is frozen and has no trained optimizer group in this sweep.

Validation is recorded every 1,048,576 tokens and at the final 5,013,504-token
state. Training records are 327,680-token aggregates. A short run is enough to
reject unstable rates and identify large optimization differences, but it is
not evidence that the selected rate is globally optimal. If candidates remain
close, extend the finalists before locking the 100M wiring study.

Run all twenty arms sequentially on the target CUDA host:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_lr_qualification \
  --skip-wire
```

Choose each mechanism's rate using final-pass K=4 validation NLL plus the
minimal finite-loss/no-catastrophic-instability guardrail. The long comparison
may use different selected rates per mechanism; retain `added_learning_rate` as
an explicit allowed difference in the long comparison and report the qualification budget.

The 2,048-token qualification supersedes the earlier 1,024-token rate sweep.
The checked-in 100M rates remain provisional until the new qualification is
run.
