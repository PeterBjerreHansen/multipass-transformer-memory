# Benchmark studies

The benchmark tree has four active roles:

- `controls/`: reusable substrate checks and vanilla smoke tests;
- `development/`: planned or qualifying scientific studies;
- `core/`: reviewed, locked studies used for central paper claims;
- `efficiency/`: hardware measurements used to choose feasible execution
  settings and report compute.

Superseded protocols and retained runs live under `historical/`. Nothing in
that directory defines the current contract or participates in automatic study
discovery. `ad_hoc/` is ignored local scratch space and must not be cited as a
study.

## Current experimental contract

The active scientific program is intentionally small:

1. `development/frozen_backbone_lr_qualification/` gives each of five feedback
   mechanisms a four-value LR sweep under an equal 5M-token budget.
2. `development/frozen_backbone_comparison/` compares 100M-token learning curves
   while the pretrained backbone remains frozen. Both recurrent mergers use the
   same late memory emission rule as the three attention variants.

These studies use 2048-token blocks and parallel K=4 validation. Downstream
tasks use K=4 context prefill followed by ordinary feedback decoding. Paper
replay/BPTT execution is deleted. The old 1024-token studies and their associated
efficiency suite have been deleted rather than archived in the repo.
A future unfrozen study must start fresh and qualify per-model learning rates.

The active studies remain planned until hardware and learning-rate qualification
is complete. Promotion means moving an unchanged
study directory to `core/` and setting `status: locked` after review.

Each development or core study owns a `STUDY.yaml`, runnable arm configs, and
its `results/` directory. Active feedback arms use whole-block parallel
multipass training. The schema and comparison rules are
documented in `docs/STUDIES.md`.

Raw checkpoints and run telemetry remain ignored under `results/<arm>/`.
Compact summaries may be tracked beside them when they are needed for a paper.

Inference/evaluation rules and durable snapshot recovery are shared across
studies. Remaining additions are in [the cleanup tracker](../docs/CLEANUP_STATUS.md). The
[A6000 timing report](development/inference_efficiency/README.md) supports keeping
routine K=4 checks and limiting optional feedback evaluation to selected durable
snapshots, initially one full block per arm.
