# Cleanup review ledger

Updated 2026-09-03. This ledger preserves the original review issue numbers.
Current behavior belongs in the [documentation map](README.md).
Pending work and user decisions belong in the [development plan](DEVELOPMENT_PLAN.md).

## Completed work

Cleanup steps 1–6 removed paper replay/BPTT and the 1024-token studies.
They retained preceding-token feedback and adaptive recirculation mixing.
The active recurrent arms now share a late writer and differ only in their merger.

Training and standalone evaluation share scoring, precision and result identity.
Exact K=1 decoding preserves feedback memory.
Durable snapshots are separate from rolling recovery checkpoints.
Interrupted publication and selected-checkpoint feedback have recovery tests.

The latest frozen-study decisions use `dense_and_strided_memory_attention` for the
combined attention pathway and align all attention readers to layers 3 and 7.
The main runs enable BOS feedback at the selected milestones.
Initial-baseline evaluation and expanded diagnostics are deferred.
See the [main protocol](../benchmarks/development/frozen_backbone_comparison/README.md) for exact settings.
None of these code or documentation changes establishes GPU qualification.

## Review issue ledger

| Original issue | Resolution or remaining work |
| --- | --- |
| 1. Hidden policy differences in NLL | Shared evaluators record policy, precision, targets and subset. |
| 2. 1024/2048 and snapshot comparability | 1024 studies deleted. Snapshot recovery implemented. Unfrozen milestones remain undecided. |
| 3. Precision mismatch | Shared explicit precision handling. Target-hardware qualification remains pending. |
| 4. BOS-only packed feedback evaluation | Standalone and selected-checkpoint validation implemented, with full and aligned counts. |
| 5. LR semantics | Equal-budget per-model frozen sweep configured. Main rates remain provisional. Unfrozen tuning is separate. |
| 6. Evaluation duplication and mode interfaces | Existing evaluators share their scorer, execution context and parameter resolution. |
| 7. Serialized full validation | Multi-block batching remains optional future work. Block limit is not batch size. |
| 8. Documentation drift | Contracts consolidated and checked against code. Documentation tests cover links, examples and protocol values. |
| 9. Exact-K downstream requirement | Withdrawn. Ordinary feedback remains the default memory-model continuation. |
| 10. Incomplete snapshot publication | Atomic weights/identity publication, retry verification and recovery tests implemented. |
| 11. K=1 feedback-memory conversion | Implemented and regression-tested, including non-identity writers. |
| 12. Merger interpretation and diagnostics | Existing simple checks first. Initial baselines and configurable-depth interventions deferred. |
| 13. Directory-layout test | Discovery follows manifests. Archive checks follow retained studies, not deleted FBT files. |
| 14. Timing-tool integration after arm replacement | Current arm paths and separate recurrent IDs implemented. Production BOS timing remains pending. |
| 15. Incomplete implementation rename | Current names now reach modules, factory dispatch, feedback fields, diagnostics, tests and active YAML paths. Retired serialized names are confined to one input-compatibility adapter; a repository naming test guards the boundary. |
| 16. Separate attention dispatch and obsolete hybrid models | The three descriptive attention names now resolve to one configurable implementation. The non-memory control is Strided Self-Attention. Named legacy hybrids are deleted; optional late recurrent memory uses the shared writer and merger modules. Architecture-aware checkpoint comparison prevents silent remapping. |

## Evidence boundaries

The [A6000 report](../benchmarks/development/inference_efficiency/README.md)
retains the original four-arm timing evidence.
Its continuation costs extrapolate a short horizon and do not time the production full-block BOS evaluator.
The two new recurrent mergers were not measured there.

Historical experiment configs and results keep their recorded names and paths.
The [grilling transcript](FROZEN_WIRING_GRILL_EXCHANGE.md) and dated research notes
include superseded proposals. They are not executable specifications.

The earlier cleanup passed 471 tests with 10 MPS skips. The later rename passed
480 tests with 10 MPS skips. The directory-layout assertion now checks only the
historical studies that are actually retained; deleted FBT and efficiency
archives are not expected to exist. Current verification results belong in the
task handoff, not a perpetually current claim in this ledger.
