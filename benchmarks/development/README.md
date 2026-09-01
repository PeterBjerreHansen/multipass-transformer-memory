# Development benchmarks

This directory contains only the studies that implement or qualify the current
paper contract:

- `forward_policy_qualification/`: 100 optimizer steps comparing paper-style
  token-diagonal Recirculation BPTT with whole-block multipass training;
- `frozen_backbone_comparison/`: 20M-token controller-only curves for BPTT
  Recirculation, multipass Recirculation, dense Memory Attention, Strided Memory
  Attention, and Multiscale Memory Attention;
- `common_checkpoint_comparison/`: the proposed 100M main comparison from one
  pretrained checkpoint, with a 5M-token backbone freeze only for feedback
  arms.

Run the forward-policy and CUDA-memory qualification first. The frozen-backbone
comparison fixes microbatch 1 and accumulation 32 across all arms. Other
development studies may change those fields together for hardware fit, but
compared arms must retain the declared effective optimizer batch unless
batching is itself the qualified variable. Learning rates remain protocol
choices and must be qualified before either comparison is promoted to `core/`.

Pass-depth stability, parameter drift, and inference diagnostics are reusable
evaluation tools, not standalone development studies. Superseded Stage 0–6 and
exploratory protocols are preserved under `../historical/`.
