# Development benchmarks

The active frozen comparison and its qualification are:

- `frozen_backbone_lr_qualification/`: completed 48-arm qualification of the
  same four LR candidates for every primary mechanism and attention extension
  at each site count, with equal 5M-token budgets. Every group selected
  `added_learning_rate: 1.0e-3`;
- `frozen_backbone_comparison/`: separate one-site and two-site 100M-token
  frozen-backbone groups, followed by attention-layout and stride extensions.

These studies use 2048-token blocks and the same K=2/3 final-pass training
objective. Most arms use microbatch 8 with accumulation 4; the combined
dense-and-strided arm uses 4x8 to preserve the same 65,536-token optimizer
batch on the target GPU. K=4 whole-block NLL is the headline validation metric.
CUDA-memory preflight and per-model learning-rate qualification are complete;
run the ad-hoc injection and stride pilots before launching the long
trajectories.

The older 1024-token studies have been deleted. Paper replay/BPTT
execution is deleted. No replacement unfrozen study is configured yet: it must
start fresh from the common pretrained checkpoint, may use a brief frozen
warmup, and needs its own per-model LR qualification.

Pass-depth stability, parameter drift, and inference diagnostics are reusable
evaluation tools, not standalone development studies. Superseded Stage 0–6 and
exploratory protocols are preserved under `../historical/`.

[Remaining additions](../../docs/DEVELOPMENT_PLAN.md) are tracked separately. The inference-efficiency
directory is a diagnostic with no study manifest; it must not be counted as a
scientific comparison or removed to satisfy study-layout tests.
