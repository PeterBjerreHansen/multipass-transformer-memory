# Development benchmarks

The active frozen comparison and its qualification are:

- `frozen_backbone_lr_qualification/`: completed 48-arm qualification of four
  LR candidates across the base mechanisms and site counts, with equal 5M-token
  budgets. Every group selected `added_learning_rate: 1.0e-3`;
- `frozen_backbone_comparison/`: nested small, medium, and large 100M-token
  frozen-backbone studies. They share one protocol and differ only in model
  count. The small tier is the primary development comparison.

These studies use 2048-token blocks and the same K=2/3 final-pass training
objective. Most arms use microbatch 8 with accumulation 4; the combined
dense-and-strided arm uses 4x8 to preserve the same 65,536-token optimizer
batch on the target GPU. K=4 whole-block NLL is the headline validation metric.
CUDA-memory preflight and base-family LR qualification are complete. The fixed
layout and stride choices were informed by private exploratory checks. Launch
one frozen tier at a time from fresh pretrained weights.

The unfrozen work is split into:

- `unfrozen_lr_qualification/`: a 20M-token, six-arm vanilla backbone-LR grid;
  the selected shared rate is carried into adaptive recirculation and dense
  Memory Attention; and
- `unfrozen_scaling_core/`: a guarded five-stage study with those three core
  arms, 2.5B feedback endpoints, and one approximately 5.35B compute-matched
  vanilla trajectory over the same corpus.

`unfrozen_scaling_extensions/` reserves optional controls without delaying the
core study. The long study starts fresh in Phase B with no frozen prefix. Its
provisional shared backbone rate is `3e-4`; training remains blocked by
`learning_rates_qualified: false` until the long-artifact and preflight gates
pass.

The older 1024-token studies have been deleted. Paper replay/BPTT execution is
deleted.

Pass-depth stability, parameter drift, and inference diagnostics are reusable
evaluation tools, not standalone development studies.

[Remaining additions](../../docs/DEVELOPMENT_PLAN.md) are tracked separately. The inference-efficiency
directory is a diagnostic with no study manifest; it must not be counted as a
scientific comparison or removed to satisfy study-layout tests.
