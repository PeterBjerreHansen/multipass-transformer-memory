# Benchmark studies

Benchmark work is organized by scientific role:

- `historical/`: retained evidence from superseded campaigns.
- `controls/`: reusable vanilla substrate controls and smoke checks.
- `development/`: structured protocol studies and retained runnable configs.
- `ad_hoc/`: one-off exploratory work and disposable diagnostics.
- `core/`: larger, predeclared studies intended to establish central claims.
- `efficiency/`: engineering measurements of throughput, memory, precision, and
  feasible batch/context sizes.

Configuration is colocated with its owner. Development and core studies should
use `STUDY.yaml` to state the scientific question, runnable arms, and declared
comparison differences without duplicating execution parameters from those
configs. The schema and conventions are documented in `docs/STUDIES.md`.

Paper-era candidates remain in `development/` while their BPTT/TBPTT,
microbatch, and learning-rate choices are qualified. Moving a directory under
`core/` means its manifest and runnable protocol are ready to be locked.

For training studies, compact summaries and comparison tables are tracked under
`results/`; each arm's generated run artifacts live directly under
`results/<arm>/` and remain ignored. Efficiency JSON files are small retained
benchmark results and may be tracked directly under `benchmarks/efficiency/results/`.
