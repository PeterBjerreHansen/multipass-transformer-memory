# Backbone learning-rate sweep

This ad-hoc study is a learning-rate screening sweep. It starts fresh
optimizer/data/RNG trajectories from the completed 100M-token vanilla and Dense
Memory Attention snapshots and runs a short 10M-token continuation for each
backbone learning rate:

- `1e-6`: existing continuation rate;
- `3e-6`: primary higher-plasticity candidate;
- `1e-5`: upper stress arm.

The 10M endpoint is approximately 10,002,432 linguistic tokens (an exact
multiple of the 2,048-token training block). Select the rate using the early
validation curves and stability, then launch a separate long continuation from
the same 100M snapshot with the winning rate. These pilots are not intended as
the final capability-training runs.

The vanilla and Dense Memory Attention protocols are otherwise copied from
their completed Stage-5 runs. `init_from` loads weights only; it does not
resume the original optimizer or sampler state. Each arm uses the same staged
Dolmino artifact and writes a new trajectory under `results/`.

The configs target the local Mac's MPS backend using FP32 compute. Validate
them locally with:

```bash
PYTHONPATH=src uv run python scripts/run_study.py \
  --study-dir benchmarks/ad_hoc/backbone_lr_sweep \
  --wire-only --wire-device mps
```

Run the complete sweep locally with:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/ad_hoc/backbone_lr_sweep \
  --arm vanilla_lr1e6 --arm vanilla_lr3e6 --arm vanilla_lr1e5 \
  --arm dense_lr1e6 --arm dense_lr3e6 --arm dense_lr1e5
```

For an interruptible CUDA cloud run, make a separate CUDA copy of the configs;
do not use the pilot outputs as the starting point for the long run. Initialize
the winner again from the corresponding completed 100M snapshot so the chosen
LR is the only changed variable.
