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

1. `development/forward_policy_qualification/` qualifies paper-style
   token-diagonal BPTT/TBPTT against whole-block multipass Recirculation.
2. `development/frozen_backbone_comparison/` compares 20M-token learning curves
   while the pretrained backbone remains frozen throughout.
3. `development/common_checkpoint_comparison/` is the proposed main 100M-token
   comparison. Every arm starts from one pretrained checkpoint; feedback arms
   freeze the pretrained backbone for the first 5M input tokens, while vanilla
   trains it from token zero.

The first study selects feasible BPTT/TBPTT and hardware settings. The latter
two remain planned until their microbatch, gradient-accumulation, learning-rate,
and truncation choices have been qualified. Promotion means moving an unchanged
study directory to `core/` and setting `status: locked` after review.

Each development or core study owns a `STUDY.yaml`, runnable arm configs, and
its `results/` directory. Arm names use `bptt` for token-diagonal recurrence and
`multipass` for whole-block parallel passes. The schema and comparison rules are
documented in `docs/STUDIES.md`.

Raw checkpoints and run telemetry remain ignored under `results/<arm>/`.
Compact summaries may be tracked beside them when they are needed for a paper.
