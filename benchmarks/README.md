# Benchmark studies

The benchmark tree has four active roles:

- `controls/`: reusable substrate checks and vanilla smoke tests;
- `development/`: planned or qualifying scientific studies;
- `core/`: reviewed, locked studies used for central paper claims;
- `efficiency/`: hardware measurements used to choose feasible execution
  settings and report compute.

Raw checkpoints and telemetry are local artifacts under `results/<arm>/` and
are ignored by Git. `ad_hoc/` is ignored local scratch space and must not be
cited as a study.

## Studies and contracts

The [development index](development/README.md) links the frozen and unfrozen
studies. Each study owns its settings, status, launch requirements and results.
The [research plan](../docs/RESEARCH_PLAN.md) defines the comparison questions;
[training](../docs/TRAINING.md) and [evaluation](../evaluation/README.md) define
shared execution semantics.

Each development or core study owns a `STUDY.yaml`, runnable arm configs, and
its `results/` directory. An arm config may use one relative `extends` path to
inherit a shared YAML fragment; child fields override inherited fields. Shared
fragments use `.yml`, while runnable arm configs use `.yaml` and must appear in
the manifest. Absolute parents and inheritance cycles are rejected.

Raw checkpoints and telemetry remain ignored under `results/<arm>/`. Keep compact
summaries beside them when they are needed to interpret results. Do not move a
live or resumable trajectory.

## `STUDY.yaml`

Required top-level fields are:

- `name`: must match the study directory name;
- `status`: one of `planned`, `active`, `complete`, or `locked`;
- `question`: the scientific question the study answers;
- `arms`: runnable arm IDs and their colocated config paths;
- `comparisons`: groups of arms that are intended to be directly comparable.

An optional `stages` list gives each named stage an exact target for every arm.
Targets must increase, be attainable whole-microbatch boundaries, appear in
each arm's `snapshot_at_tokens`, and end at that arm's configured
`max_unique_tokens`. Different arms may therefore stop at different token
presentations for compute matching.

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

Use `run_study.py --stage <name>` or `run-cloud-study --stage <name>` to stop
each arm at its declared target. The final horizon remains in the config from
the first launch, so LR schedules and resume compatibility do not change.

`make check` includes this gate.

## Results and generated artifacts

Keep small result tables, JSON summaries, and interpretation notes directly in
`results/` when they are worth retaining. Raw checkpoints, `run.json`,
`metrics.jsonl`, and other execution artifacts belong under the relevant
`results/<arm>/` directory and are ignored by Git.

This preserves locality without turning the source tree into a checkpoint
archive. When a generated checkpoint initializes another run, the trainer
records its SHA-256 in the new run's provenance.
