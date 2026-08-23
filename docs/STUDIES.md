# Study organization

Scientific configuration is colocated with the study that owns it. The
repository intentionally has no central `configs/` tree.

A development or core study has the following shape:

```text
benchmarks/<role>/<study>/
  README.md
  STUDY.yaml
  <arm>.yaml
  results/
    README.md
    <retained summaries>
    <arm>/                  # ignored raw run artifacts
```

The runnable arm YAML is the authoritative execution specification. It should
state every scientifically relevant setting explicitly. `STUDY.yaml` describes
the comparison rather than copying token budgets, learning rates, precision, or
other execution values.

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
    experimental_axes: [pass_schedule, pass_loss_weights]
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
