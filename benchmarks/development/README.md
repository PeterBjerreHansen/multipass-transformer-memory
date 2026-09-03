# Development benchmarks

The active frozen comparison and its qualification are:

- `frozen_backbone_lr_qualification/`: four LR candidates for each of five
  mechanisms, with equal 5M-token budgets;
- `frozen_backbone_comparison/`: 100M-token frozen-backbone curves for two
  late-memory recurrent mergers (projected residual and adaptive recirculation),
  dense Memory Attention, Strided Memory Attention, and Multiscale Memory Attention.

These studies use 2048-token blocks, microbatch 8 with accumulation 4, and the
same K=2/3 final-pass training objective. K=4 whole-block NLL is the headline
validation metric. Run CUDA-memory preflight and qualify per-model learning rates
before launching the long trajectories.

The older 1024-token studies have been deleted. Paper replay/BPTT
execution is deleted. No replacement unfrozen study is configured yet: it must
start fresh from the common pretrained checkpoint, may use a brief frozen
warmup, and needs its own per-model LR qualification.

Pass-depth stability, parameter drift, and inference diagnostics are reusable
evaluation tools, not standalone development studies. Superseded Stage 0–6 and
exploratory protocols are preserved under `../historical/`.

[Remaining additions](../../docs/CLEANUP_STATUS.md) are tracked separately. The inference-efficiency
directory is a diagnostic with no study manifest; it must not be counted as a
scientific comparison or removed to satisfy study-layout tests.
