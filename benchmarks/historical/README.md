# Historical benchmarks

This directory preserves superseded protocols and locally retained run
artifacts for provenance. Its contents are read-only evidence, not active
experiment definitions. Automatic study discovery intentionally scans only
`benchmarks/development/` and `benchmarks/core/`.

- `staged_pipeline/` contains the former Stage 0–6 workflow, including the
  completed 100M Stage-5 campaign and the partial/continued Stage-6 work.
- `exploratory/` retains the backbone learning-rate sweep.

The separate exploratory FBT study and historical efficiency directory have
been removed from this checkout. The compatibility implementations remain.

Files inside these studies retain their original paths and commands. Those
references document how the runs were made and are not expected to resolve
after archival. The former pilot, staged-100M, 2.5B continuation, and Stage-6
data recipes were also deleted from `data/dolmino/`; historical configs that
mention them are therefore provenance records, not runnable data preparation
commands. Do not edit a historical config to make it runnable in the new
layout; copy it into a new active study if a follow-up experiment is needed.

Large `results/<arm>/` payloads remain ignored by Git but are kept locally in
this tree. Moving them here does not make them suitable for source control or a
paper artifact release.

## Deleted 1024-token studies

The former forward-policy, common-checkpoint and 1024-token frozen studies
have been deleted at the user's request. They are no longer archived in this
tree. The retained 2048-token staged protocols remain unchanged. Paper
readout/replay and its BPTT/TBPTT implementation are also deleted.
