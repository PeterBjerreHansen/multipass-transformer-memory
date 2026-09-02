# Development benchmarks

This directory contains only the studies that implement or qualify the current
paper contract:

- `forward_policy_qualification/`: 100 optimizer steps comparing window-128
  token-diagonal Recirculation TBPTT with whole-block multipass training;
- `frozen_backbone_comparison/`: 20M-token controller-only curves for multipass
  Recirculation, dense Memory Attention, Strided Memory Attention, and
  Multiscale Memory Attention. The token-diagonal TBPTT policy is kept as an
  optional standalone configuration inside that directory;
- `common_checkpoint_comparison/`: the proposed 100M main comparison from one
  pretrained checkpoint, with a 5M-token backbone freeze only for feedback
  arms.

Run the forward-policy and CUDA-memory qualification first. Active studies fix
the effective optimizer batch at 32 sequences and use one physical batching
policy across every arm in a study. The frozen-backbone and forward-policy
studies use microbatch 16 with accumulation 2. The fully trainable
common-checkpoint study uses microbatch 8 with accumulation 4. Learning rates
remain protocol choices and must be qualified before either comparison is
promoted to `core/`.

Pass-depth stability, parameter drift, and inference diagnostics are reusable
evaluation tools, not standalone development studies. Superseded Stage 0–6 and
exploratory protocols are preserved under `../historical/`.
