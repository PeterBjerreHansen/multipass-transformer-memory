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
is complete. Promotion means moving the reviewed
study definition to `core/` and setting `status: locked` after review.
Before its first retained run, update each `output_dir` to the new arm-local
path and verify the manifest. Do not move a live or resumable trajectory.

Each development or core study owns a `STUDY.yaml`, runnable arm configs, and
its `results/` directory. Active feedback arms use whole-block parallel
multipass training. The schema and comparison rules are
defined below.

Raw checkpoints and run telemetry remain ignored under `results/<arm>/`.
Compact summaries may be tracked beside them when they are needed for a paper.

Inference/evaluation rules and durable snapshot recovery are shared across
studies. Next work is in [the development plan](../docs/DEVELOPMENT_PLAN.md). The
[A6000 timing report](development/inference_efficiency/README.md) supports keeping
routine K=4 checks and limiting optional feedback evaluation to selected durable
snapshots, initially one full block per arm.

## `STUDY.yaml`

Required top-level fields are:

- `name`: must match the study directory name;
- `status`: one of `planned`, `active`, `complete`, or `locked`;
- `question`: the scientific question the study answers;
- `arms`: runnable arm IDs and their colocated config paths;
- `comparisons`: groups of arms that are intended to be directly comparable.

Core studies must use `status: locked`. A non-planned study must declare at
least one runnable arm.

Example:

```yaml
name: k_selection
status: complete
question: Does K=3 justify its extra compute for adaptive Recirculation?
arms:
  - {id: recirculation_k2, config: recirculation_k2.yaml}
  - {id: recirculation_k3, config: recirculation_k3.yaml}
comparisons:
  - name: recirculation_k
    arms: [recirculation_k2, recirculation_k3]
    experimental_axes: [pass_schedule, ntp_pass_loss_weights]
```

`experimental_axes` names config fields that intentionally vary as part of the
scientific comparison. `allowed_differences` is available for a field that must
vary for a non-scientific reason. For example, a cross-architecture comparison
might need architecture-specific initialization paths. Use this escape hatch
sparingly and document why the difference is necessary in the study README.

`output_dir` is always arm-local and therefore excluded automatically. Other
fields, including `init_from` and `resume_from`, must match unless explicitly
declared. This prevents an apparently controlled comparison from silently using
different data, optimization, precision, initialization, or evaluation
settings. In particular, `batch_size` and `grad_accum_steps` are not treated as
mere hardware metadata: together with sequence length they determine the
optimizer-batch tokens and must remain controlled unless the study explicitly
asks about batching.

## Verification

Run:

```bash
uv run python scripts/verify_study.py
```

with no arguments to validate every development/core study, or pass one or more
study directories explicitly. The verifier checks that:

- manifests are structurally valid;
- all declared arm configs exist and parse;
- no runnable config is silently omitted from the manifest;
- each arm writes under its own `results/<arm>/` path;
- compared arms differ only on declared fields;
- core studies are locked before execution.

`make check` includes this gate.

## Results and generated artifacts

Keep small result tables, JSON summaries, and interpretation notes directly in
`results/` when they are worth retaining. Raw checkpoints, `run.json`,
`metrics.jsonl`, and other execution artifacts belong under the relevant
`results/<arm>/` directory and are ignored by Git.

This preserves locality without turning the source tree into a checkpoint
archive. When a generated checkpoint initializes another run, the trainer
records its SHA-256 in the new run's provenance.
