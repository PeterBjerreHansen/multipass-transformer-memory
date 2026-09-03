# Historical benchmarks

This directory preserves superseded protocols and locally retained run
artifacts for provenance. Its contents are read-only evidence, not active
experiment definitions. Automatic study discovery intentionally scans only
`benchmarks/development/` and `benchmarks/core/`.

- `staged_pipeline/` contains the former Stage 0–6 workflow, including the
  completed 100M Stage-5 campaign and the partial/continued Stage-6 work.
- `exploratory/` contains FBT investigations, learning-rate sweeps, and related
  pilots.
- `efficiency/` contains suites and measurements tied to the superseded
  architecture set and 2,048-token protocol.

Files inside these studies retain their original paths and commands. Those
references document how the runs were made and are not expected to resolve
after archival. Do not edit a historical config to make it runnable in the new
layout; copy it into a new active study if a follow-up experiment is needed.

Large `results/<arm>/` payloads remain ignored by Git but are kept locally in
this tree. Moving them here does not make them suitable for source control or a
paper artifact release.

## Deleted 1024-token studies

The former forward-policy, common-checkpoint and 1024-token frozen studies
have been deleted at the user's request. They are no longer archived in this
tree. The separate 2048-token historical protocols remain unchanged. Paper
readout/replay and its BPTT/TBPTT implementation are also deleted.
