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

These studies use 2048-token blocks and parallel K=4 validation. The nominal
optimizer batch is 65,536 tokens per update: standard arms use batch 8 with
four accumulation steps, while the memory-heavy
`dense_and_strided_memory_attention` arm uses batch 4 with eight accumulation
steps. Downstream
tasks use K=4 context prefill followed by ordinary feedback decoding. Paper
replay/BPTT execution is deleted. The old 1024-token studies and their associated
efficiency suite have been deleted rather than archived in the repo.
The active data inputs are `data/dolmino/gpu_2048` for these studies and
`data/dolmino/wiring_2048` for wiring/pre-training checks. Former staged and
long-run data recipes were retired with the clean-slate reset.
A future unfrozen study must start fresh and qualify per-model learning rates.

The previous GPU qualification and comparison trajectories were discarded after
the tokenizer-padding audit. The clean artifact must be regenerated and the
qualification restarted before any frozen result is promoted. Promotion means
moving the reviewed study definition to `core/` and setting `status: locked`
after review.
Each active `STUDY.yaml` pins the regenerated data manifest hash; the study
runner verifies that exact artifact before wiring or training. The 100M
comparison remains blocked until its per-model learning rates are qualified.
Before its first retained run, update each `output_dir` to the new arm-local
path and verify the manifest. Do not move a live or resumable trajectory.

Each development or core study owns a `STUDY.yaml`, runnable arm configs, and
its `results/` directory. Active feedback arms use whole-block parallel
multipass training. The schema and comparison rules are
defined below.

An arm config may use one relative `extends` path to inherit a shared YAML
fragment; child fields override inherited fields. Shared fragments use the
`.yml` suffix, while runnable arm configs use `.yaml` and must appear in the
study manifest. Absolute parents and inheritance cycles are rejected.

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

Before execution, every non-planned study must declare `data_artifacts`, mapping
every arm's `data_dir` to the exact SHA-256 of its verified `manifest.json`.
Planned studies may omit the mapping while data materialization is still
pending. A study whose final rates still await qualification sets
`learning_rates_qualified: false`; the runner then permits wiring-only checks
but refuses training.

Example:

```yaml
name: k_selection
status: complete
question: Does K=3 justify its extra compute for adaptive Recirculation?
data_artifacts:
  data/dolmino/wiring_2048: <64-character manifest SHA-256>
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
- core studies are locked before execution;
- active and locked studies declare data pins before execution;
- declared data pins cover exactly the data directories used by the arms.

At execution time, `run_study.py` verifies the complete artifact, checks its
manifest against the pinned hash, and enforces the learning-rate gate.
`run-cloud-study` enforces the same gate and makes the remote launcher check the
remote manifest hash before it starts training.

`make check` includes this gate.

## Results and generated artifacts

Keep small result tables, JSON summaries, and interpretation notes directly in
`results/` when they are worth retaining. Raw checkpoints, `run.json`,
`metrics.jsonl`, and other execution artifacts belong under the relevant
`results/<arm>/` directory and are ignored by Git.

This preserves locality without turning the source tree into a checkpoint
archive. When a generated checkpoint initializes another run, the trainer
records its SHA-256 in the new run's provenance.
