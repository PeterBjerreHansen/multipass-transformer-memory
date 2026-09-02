# Frozen-backbone learning-rate qualification

This planned qualification selects the Phase-A learning rate for all four
feedback mechanisms before interpreting the frozen-backbone wiring comparison.
The backbone remains frozen throughout each run; only the added controller or
Memory Attention parameters are optimized.

## Protocol

Every arm consumes exactly 5,013,504 unique linguistic tokens, or 153 optimizer
updates with the common microbatch-16/accumulation-2 policy. The pass schedule
is the paper wiring schedule: K=2 with probability 0.9 and K=3 with probability
0.1, with final-pass-only NTP loss weights. Validation uses deterministic K=4
whole-block pass-depth NLL on 64 held-out blocks; generation is not part of this
qualification.

The candidate grid is the same for every mechanism:

```text
3e-5, 1e-4, 3e-4
```

Validation is recorded every 1,048,576 tokens and at the final 5,013,504-token
state. Training records are 327,680-token aggregates. A short run is enough to
reject unstable rates and identify large optimization differences, but it is
not evidence that the selected rate is globally optimal. If candidates remain
close, extend the finalists before locking the 100M wiring study.

Run all twelve arms sequentially on the target CUDA host:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_lr_qualification \
  --skip-wire
```

Choose the rate using final-pass K=4 validation NLL, trajectory stability,
gradient/parameter-drift diagnostics, and measured training time. For a clean
architecture comparison, use one common stable rate across all four variants.
If rates are selected independently, retain `added_learning_rate` as an
explicit experimental axis and report the qualification budget.

The matched-grid qualification supersedes the earlier architecture-specific
rate choices. The active 100M frozen-backbone wiring study uses `3e-4` for all
four mechanisms.
