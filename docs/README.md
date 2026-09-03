# Documentation map

Use this map for current behavior. Runnable configs define experiment settings.
Historical records and research notes do not override those configs or the current contracts.

## Authoritative guides

| Question | Read |
| --- | --- |
| Which experiments run, and how are studies organized? | [Benchmark protocols and manifest schema](../benchmarks/README.md) |
| What runs next, and which decisions are deferred? | [Development plan](DEVELOPMENT_PLAN.md) |
| Which model families are active or historical? | [Architecture map](ARCHITECTURES.md) |
| How do the recurrent mergers work? | [Recurrent memory](RECURRENT_MEMORY.md) |
| How do attention memory, retention and MEM slots work? | [Memory Attention](MEMORY_ATTENTION.md) |
| How are packed data and splits constructed? | [Data contract](DATA.md) |
| How do training, resume and snapshots work? | [Training contract](TRAINING.md) |
| What do NLL, feedback, precision and downstream results mean? | [Evaluation contract](../evaluation/README.md) |
| How do exact cached and feedback decoding differ? | [Inference contract](RECURRENT_INFERENCE.md) |
| How do we preflight and operate a cloud run? | [Cloud runbook](CLOUD.md) |
| Which correctness checks must pass? | [Validation gates](VALIDATION.md) |
| Which command exposes each operation? | [Script index](../scripts/README.md) |
| Where did the backbone, data and mechanisms come from? | [Provenance](UPSTREAMS.md) |

Architecture contracts define mechanisms. Study READMEs explain their selected settings.
The evaluation guide owns scoring and precision. The training guide owns scheduling and recovery.
Other pages should link to these definitions instead of copying them.

## Historical and decision records

- [Cleanup ledger](CLEANUP_STATUS.md): original review issues and their resolution.
- [Grilling exchange](FROZEN_WIRING_GRILL_EXCHANGE.md): the original discussion, including superseded proposals.
- [Recirculation indexing](research/recirculation-token-indexing.md) and
  [feedback-merger research](research/looped-feedback-mechanisms.md): dated research and earlier recommendations.
- [Archived experiments](../benchmarks/historical/README.md): original protocols, paths and scientific caveats.

Archived commands are provenance, not instructions for launching the current campaign.
Do not rewrite historical results to match new model names or reader layouts.

## Maintaining the docs

Verify behavior against the implementation before changing a contract.
Update the owning study when a default changes. Update the development plan when a task is deferred or completed.
Run `make check` after edits. Documentation tests check local links, examples and selected config-backed protocol values.
