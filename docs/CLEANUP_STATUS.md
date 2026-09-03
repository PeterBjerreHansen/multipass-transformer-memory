# Cleanup status and remaining work

Updated 2026-09-03. This is the handoff for the agreed cleanup, not permission
to implement later steps or launch training. Preserve the original review issue
numbers below when closing work.

## Completed cleanup: steps 1–6

- Removed executable paper readout/replay, `paper_recirculation` inference and
  validation, and its BPTT/TBPTT training, truncation and checkpointing branches.
- Removed the corresponding exports, CLI choices, FLOP estimates, efficiency
  cases, launcher targets and positive tests for that implementation.
- Retained the adaptive recirculation merger, shared late-memory writer,
  preceding-token multipass computation and ordinary feedback decoding.
- Deleted the 1024-token forward-policy, common-checkpoint and older frozen
  studies, their associated efficiency suite and the paper_1024 preparation
  recipe, following the user's follow-up. They are no longer archived in-repo.
  The older frozen study's local logs/metadata were moved to a temporary
  recovery directory outside the repo; the task handoff records its location.
  No 2048-token historical study or active run artifacts were removed.
- Active study discovery now contains the frozen comparison and its LR sweep;
  diagnostics without a `STUDY.yaml` do not count as scientific studies.
- Removed-policy configs and checkpoints fail explicitly. Old multipass
  metadata containing only the unused `false`/`null` defaults still loads;
  those obsolete fields are not serialized into new experiment configs.

No new unfrozen study, inference algorithm or training run was introduced.
An opt-in selected-checkpoint feedback schedule is now available, but the current
frozen configs and their per-model LR grid are unchanged and do not enable it.

Verification: 469 tests passed and 10 MPS-dependent tests skipped (24.72 seconds
with CPU thread counts limited to one). Both active
study manifests validate; byte-compilation and `git diff --check` pass. The CLI
regressions exercise checkpoint identity, default resolution, precision overrides
and downstream scoring counts. Added coverage includes full 2048-position BOS
feedback, precision dispatch, actual mid-decode signals, snapshot/report recovery,
retention, unchanged training state and final-validation de-duplication.
No GPU qualification or new hardware timing run was performed.

## Inference contract to preserve

| Use | Initialization | Computation |
| --- | --- | --- |
| Routine packed NLL | Full 2048-token block | Parallel K=4; retain K=1–4 results |
| Pure-feedback NLL | BOS-only prefill, K=1 | Observed tokens through ordinary feedback decoding |
| Downstream tasks | Actual context, K=4 prefill | Feedback continuation |
| Standard control | Ordinary context | Standard decoding |

BOS-only is an input choice, not a new inference mode. It is now integrated into
packed-text evaluation and optional selected-checkpoint validation. The existing
continuation diagnostic uses a prefix from the data: `prompt_tokens=1` is not
automatically a BOS token. Exact cached K-pass decoding remains a correctness
reference/optional diagnostic, not the downstream default.

## Cleanup 3 — inference (implemented)

- Prefill depth and continuation policy remain independent. BOS-only and
  contextual prompts reuse ordinary feedback decoding.
- K=1 exact steps now update architecture-specific feedback memory. Conversion
  to feedback is tested against fresh prefill with both recurrent mergers and
  non-identity writers.
- The same-checkpoint diagnostic now reports `standard_k1_nll`,
  `standard_k1_nll_by_offset`, and `recurrent_minus_standard_k1`, not misleading
  `vanilla` labels. Historical reports are unchanged; new CLI output has schema 2.
- Added BOS/context and strict-past regression coverage. Merger definitions,
  injection placement and the downstream K=4-prefill/feedback default are unchanged.

## Cleanup 4 — evaluation (implemented)

- Trainer validation, final-pass NLL and pass-depth evaluation share one
  parallel loop, model-owned labels, token-weighted per-source/per-pass scoring
  and subset metadata.
- Continuation diagnostics, downstream scoring/generation and single-transition
  interventions share the precision context and scorer. Standard/cached
  downstream paths also share truncation. Model mode is restored after success
  or failure.
- CLI defaults come from experiment `eval_passes`, optional
  `eval_prefill_passes` and `eval_decode_mode`, and `autocast_dtype`.
  Explicit flags override them without modifying checkpoint compatibility
  settings. No named policy profiles were added.
- Results identify actual precision, computation policy, checkpoint (or live
  trainer state), split/subset and target counts. Continuation reports include
  offset/horizon counts; packed evaluators include per-source counts; downstream
  output includes per-task scoring-operation counts and raw harness evidence.
- Existing parallel scored positions are preserved, including input-only MEM slots.
  Its first packed token remains unscored. Data-prefix continuation is explicitly
  distinct from BOS initialization. No old FP32 result is relabelled BF16.
- Multi-block batching and depth-aligned K=4 interventions
  remain later additions. Standalone block limits still default to the full
  split; use `--max-blocks 64` for routine-check parity and `1` for the initial
  expensive diagnostic.

See [evaluation/README.md](../evaluation/README.md) for the resolved contract
and [PRECISION.md](PRECISION.md) for explicit compute overrides.

## Cleanup 5 — durable planned snapshots (implemented)

- Preserve rolling resumable checkpoints separately from planned weights snapshots.
- Commit a resumable checkpoint with pending snapshot work before publishing
  weights. Recover that work before training advances, including at run end.
- Embed config and identity with weights in one atomic safetensors publication.
  Repair the sidecar on retry. Verify committed weights and coalesce thresholds
  crossed by the same optimizer update.
- Test interrupted writes, post-rename failure, missing sidecars, portable
  loading, conflicting weights, repeat recovery and independent retention.

## Cleanup 6 — documentation reconciliation (implemented)

README now leads with the active experiment. Historical results retain their
original scope and caveats. Training, evaluation, inference, validation, cloud
and study docs agree on K=4 routine/contextual evaluation, previous-token
feedback, fresh unfrozen initialization and independent checkpoint/snapshot
roles. The original exchange and historical evidence remain unchanged.

## A6000 evaluation-cost guidance

The [Verda qualification](../benchmarks/development/inference_efficiency/README.md)
contains local FP32 and BF16 reports. FP32 K=4 costs were about 16–18 seconds
per 64-block check and 4–5 minutes for the roughly 2M-token split. The combined
exact-vs-feedback diagnostic projects to 4–5 minutes per block and 65–75 hours
for the split. These are not feedback-only timings: the report separately
projects feedback-only continuation to roughly 40–48 seconds per block.

Continuation costs were measured through 128 tokens and extrapolated; a full
2048-token BOS-only evaluation was not directly timed. The four tested variants
include the older middle-layer recirculation, not the two new recurrent mergers.
BF16 sped up full-block validation but slowed cached feedback on this setup.
Do not assume either precision wins for every workload or silently change it.

Retain routine K=4 checks every 3,276,800 tokens. Optional feedback evaluation
pauses training only at selected durable snapshots, outside optimizer timing.
Start with one fixed full block per arm and expand only when useful separation
appears. A one-block result is not a statistically reliable ranking.

## Critical cleanup and BOS feedback addition

- Snapshot metadata with an unsupported format fails closed; retries cannot
  silently overwrite it. Directory-fsync logic is shared across training files.
- A completed routine validation is reused at the same final checkpoint only
  when depth, forward policy, precision and block count still match. Changed
  evaluation settings are not mistaken for completed work.
- `evaluate_nll.py --forward feedback` uses ordinary feedback with BOS-only K=1
  prefill, teacher-forcing the complete block without an exact-reference stream.
  Model-owned labels preserve MEM prediction positions. Full and aligned scores
  explicitly report 2048 and 2047 targets per ordinary block.
- `feedback_eval_at_tokens` selects planned snapshot thresholds;
  `feedback_eval_max_blocks` defaults to one; `feedback_eval_autocast_dtype`
  permits an explicit precision choice. Default behavior remains disabled.
- Snapshot-bound reports are atomic, identified by weights/source/runtime/data/settings,
  and retained separately from rolling checkpoints. Interrupted checks retry
  before training advances; completed reports and journal events are idempotent.
  Routine K=4 validation and early stopping remain separate.

## Review issue ledger

| Original issue | Status / next owner step |
| --- | --- |
| 1. Hidden policy differences in NLL | Closed in cleanup 4: policy, precision, targets and subset recorded |
| 2. 1024/2048 and snapshot comparability | 1024 studies deleted; snapshot recovery fixed; future unfrozen milestones still to specify |
| 3. Precision mismatch | Closed in cleanup 4; hardware qualification remains separate |
| 4. BOS-only packed feedback evaluation | Implemented: standalone and selected-checkpoint validation, full/aligned counts |
| 5. LR semantics | Frozen manifest supports per-model selection; fresh unfrozen qualification remains later work |
| 6. Evaluation duplication / mode interfaces | Closed in cleanup 3–4 for existing evaluators; later additions still separate |
| 7. Serialized full validation | Later batching addition; distinguish block limit from batch size |
| 8. Documentation drift | Cross-document reconciliation completed in cleanup 6 |
| 9. Exact-K downstream requirement | Withdrawn: ordinary feedback remains the intended continuation |
| 10. Incomplete snapshot publication | Fixed and regression-tested in cleanup 5 |
| 11. K=1 feedback-memory conversion | Fixed and regression-tested in cleanup 3 |
| 12. Merger interpretation / diagnostics | Preserve current architecture; add initial K=4 and depth-aligned memory-use diagnostics later |
| 13. Directory-layout test | Fixed: test manifests, not all development directories |

## Remaining additions

Remaining additions are evaluation batching,
initial-state/K=4 memory interventions, and a fresh 2048-token unfrozen study
with per-model backbone/added-parameter LR sweeps and aligned snapshots. The
unfrozen run may begin briefly frozen, but must not continue the frozen study.
Review loss weights explicitly. Do not add more merger candidates at this stage.
