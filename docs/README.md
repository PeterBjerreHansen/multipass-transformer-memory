# Documentation map

Use this map for current behavior. Runnable configs define experiment settings.
Research notes and decision records do not override those configs or the current contracts.

## Authoritative guides

| Question | Read |
| --- | --- |
| What is the scientific thesis and comparison contract? | [Research plan](RESEARCH_PLAN.md) |
| Which experiments run, and how are studies organized? | [Benchmark protocols and manifest schema](../benchmarks/README.md) |
| In what order should the agreed work be implemented and verified? | [Staged implementation plan](DEVELOPMENT_PLAN.md) |
| Which model families are active or historical? | [Architecture map](ARCHITECTURES.md) |
| How do the recurrent mergers work? | [Recurrent memory](RECURRENT_MEMORY.md) |
| How do attention memory and retention work? | [Memory Attention](MEMORY_ATTENTION.md) |
| How are packed data and splits constructed? | [Data contract](DATA.md) |
| How do training, resume and snapshots work? | [Training contract](TRAINING.md) |
| What do NLL, feedback, precision and downstream results mean? | [Evaluation contract](../evaluation/README.md) |
| How do exact cached and feedback decoding differ? | [Inference contract](FEEDBACK_INFERENCE.md) |
| How do we preflight and operate a cloud run? | [Cloud runbook](CLOUD.md) |
| Which correctness checks must pass? | [Validation gates](VALIDATION.md) |
| Which command exposes each operation? | [Script index](../scripts/README.md) |
| Where did the backbone, data and mechanisms come from? | [Provenance](UPSTREAMS.md) |

Architecture contracts define mechanisms. Study READMEs explain their selected settings.
The evaluation guide owns scoring and precision. The training guide owns scheduling and recovery.
Other pages should link to these definitions instead of copying them.

## Decision records

- [Cleanup ledger](CLEANUP_STATUS.md): original review issues and their resolution.
- [Recirculation indexing](research/recirculation-token-indexing.md) and
  [feedback-merger research](research/looped-feedback-mechanisms.md): dated research and earlier recommendations.
- [Unfrozen benchmark decision record](research/unfrozen-benchmark-plan-audit.md):
  the current three-arm, staged scaling rationale and launch gates.

Research notes are context, not instructions for launching the current campaign.

## Maintaining the docs

Verify behavior against the implementation before changing a contract.
Update the owning study when a default changes. Update the development plan when a task is deferred or completed.
Run `make check` after edits. Documentation tests check local links, examples and selected config-backed protocol values.
