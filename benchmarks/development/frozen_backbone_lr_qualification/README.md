# Frozen-backbone learning-rate qualification

This planned qualification selects the Phase-A learning rate for all four
feedback mechanisms before interpreting the frozen-backbone wiring comparison.
The backbone remains frozen throughout each run; only the added controller or
Memory Attention parameters are optimized. This qualification uses the same
2,048-token `gpu_2048` artifact and 8x4 default optimizer batching as the
comparison study.

## Protocol

Every arm consumes exactly 5,013,504 unique linguistic tokens. With 2,048-token
blocks and the nominal 8x4 policy this is 77 optimizer updates, with the final
update using the remaining partial accumulation. The pass schedule
is the paper wiring schedule: K=2 with probability 0.9 and K=3 with probability
0.1, with final-pass-only NTP loss weights. Validation uses deterministic K=4
whole-block pass-depth NLL on 64 held-out blocks; generation is not part of this
qualification.

The candidate grid is the same for every mechanism:

```text
3e-5, 1e-4, 3e-4, 1e-3
```

Validation is recorded every 1,048,576 tokens and at the final 5,013,504-token
state. Training records are 327,680-token aggregates. A short run is enough to
reject unstable rates and identify large optimization differences, but it is
not evidence that the selected rate is globally optimal. If candidates remain
close, extend the finalists before locking the 100M wiring study.

Run all sixteen arms sequentially on the target CUDA host:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_lr_qualification \
  --skip-wire
```

Choose each mechanism's rate using final-pass K=4 validation NLL plus the
minimal finite-loss/no-catastrophic-instability guardrail. The long comparison
may use different selected rates per mechanism; retain `added_learning_rate` as
an explicit experimental axis and report the qualification budget.

The 2,048-token qualification supersedes the earlier 1,024-token rate sweep.
The checked-in 100M rates remain provisional until the new qualification is
run.
