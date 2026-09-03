# Development plan

Updated 2026-09-03. This plan records the agreed frozen-study settings and next
tasks. It does not authorize paid runs or select an unfrozen protocol on the
user's behalf.

## Current position

The code supports the intended comparison, but the restructured five-arm study
is still planned, not GPU-qualified or scientifically concluded.

- Active arms: two late-memory recurrent mergers (`projected_residual` and
  adaptive `recirculation`), plus dense, strided and dense-and-strided Memory Attention.
- All use preceding-token/previous-pass memory. The recurrent pair shares the
  writer rule and read location. Paper replay/BPTT and the 1024 studies are gone.
- All three attention arms read at `[3, 7]`; the recurrent arms read at `[3]`.
  All descriptive attention names resolve to one `memory_attention` implementation.
  Their pattern settings and public names remain in config metadata.
  The separate non-memory control is `strided_self_attention`.
  Removed named hybrids are rejected; optional late recurrence is configured with
  `recurrent_merger` and `recurrent_layers`, without adding a study arm.
- The frozen configs use 2048-token blocks, K=2/3 final-pass training, K=4
  routine validation, and a four-rate qualification grid for each model.
  Long-run rates are still provisional.
- Shared evaluators record precision, scored targets, subset and weight identity.
  BOS-only feedback NLL is available standalone and after selected snapshots;
  it reports full and aligned target sets. All five 100M configs enable one full
  prefix block at the existing approximately 5M, 20M and 100M snapshots:
  `[5013504, 20021248, 100007936]`. The LR sweep leaves feedback evaluation off.
- Planned snapshots and rolling recovery checkpoints are separate. Publication,
  pending feedback checks and report/journal retries have recovery tests.
- The earlier A6000 reports measure four older arms and extrapolate cached
  continuations. They do not measure the new full-block BOS NLL path or both
  new recurrent mergers. Historical scientific results are not results for the
  restructured study.

The [cleanup ledger](CLEANUP_STATUS.md) records the original findings and their
resolution. Architecture and evaluation contracts remain in
[RECURRENT_MEMORY.md](RECURRENT_MEMORY.md) and
[evaluation/README.md](../evaluation/README.md); do not duplicate their definitions
in new study-specific evaluators.

## Recommended sequence

| Priority | Work package | Deliverable / completion check |
| --- | --- | --- |
| 1 | Simple checks and ablations | Existing pass-depth and single-transition real/zero/mismatched-memory checks; no new diagnostic framework or token-zero evaluation gate |
| 2 | Target-GPU pre-training check | Actual five-arm optimizer/validation fit, directly measured BOS NLL time, and an interruption/resume check before qualification training |
| 3 | Frozen per-model LR selection | The existing 20-arm qualification, a documented selection for each model, then resolved long-run configs |
| 4 | Grill the unfrozen plan, then specify and qualify it | An agreed protocol, separate study and per-model backbone/added-parameter sweep, starting from pretrained weights |
| 5 | Comparison reporting | Snapshot-aligned frozen/unfrozen reports with explicit target and computation differences |

Specify the unfrozen protocol while the short frozen qualification is underway.
Do not make it wait for completion of every 100M frozen trajectory. Its weights,
optimizer and learning-rate choices must nevertheless be independent.

### 1. Start with simple checks and ablations

Use the existing shared evaluators first. Pass-depth evaluation already accepts
arbitrary K; the active experiment retains K=1–4 NLL. At an early trained
snapshot, run `evaluate_memory_interventions.py` on a small fixed subset to
compare real, zeroed and mismatched memory for the K=1-to-K=2 transition. Use
the same targets and precision for every condition and arm. This is a simple
memory-use check, not a K=4 intervention or a separately trained ablation arm.

Check finite loss/gradients and whether training improves held-out NLL. Do not
introduce a hand-chosen quality threshold or require a new diagnostic framework
before training. Dedicated token-zero/initial-baseline evaluation is deferred
by user decision; the existing standalone option remains available. Compare
absolute trained-checkpoint NLL, without claiming improvement from an unmeasured
initial loss or assuming initialization effects have been proven to disappear.

Deeper analysis follows once training works or a simple check reveals a problem.
Keep that future work in the shared evaluation module: explicit pass depth and
transition selection, matching-depth mismatch donors, a separate merger-bypass
control, and optional writer/merger scales. K=4 must remain an experiment default,
not an implementation limit. Zero memory is not merger bypass: adaptive
recirculation can still rescale the destination with zero source.

Live-feedback interventions are also deferred. If added, record prompt kind,
prefill depth, continuation horizon and intervention timing independently;
parallel-pass semantics must not be applied to feedback continuation silently.

### 2. Pre-training check on the actual execution path

Follow the [cloud pre-training checklist](CLOUD.md#pre-training-checks) before LR qualification.
It includes real optimizer updates, validation, production BOS timing and snapshot/resume.
The existing forward/backward-only check is insufficient.

Keep preflight outputs separate. Do not reuse their weights for scientific runs.
The production BOS evaluator still needs a timing case in the efficiency script.
Do not substitute synthetic cached-continuation extrapolations for that measurement.

The main-run feedback schedule is already selected in the
[frozen protocol](../benchmarks/development/frozen_backbone_comparison/README.md).
Precision currently inherits BF16. Keep any subsequently agreed precision change
explicit and common across arms. No full-split feedback campaign is implied.
Unfrozen hardware checks follow only after that separate protocol is agreed.

### 3. Finish frozen qualification and run the comparison

Use the existing five-model, four-rate, 5,013,504-token qualification. Select each
model's added-parameter LR using K=4 validation NLL and basic stability evidence,
recording the equal tuning budget. Do not reuse old middle-layer/1024 selections
as qualification of the new recurrent arms.

Update the five long-run configs with the selected rates, retaining the agreed
feedback milestones. Keep one trajectory per arm, durable snapshots, the same data and
routine target set. If LR candidates remain indistinguishable, decide whether a
small finalist extension is worth its cost; do not silently expand one model's
search budget. An individual 5M qualification trajectory is not automatically
the first segment of the final 100M run.

### 4. Define the fresh unfrozen experiment

**Required pre-experiment review: hold a "grill me" session with the user before
locking the unfrozen protocol or launching its qualification/long trajectories.**
Status: pending; do not treat the decision list below as already settled.
Use the grilling skill's dependency-ordered question rounds: establish facts
from the repo, ask the currently decidable questions with recommendations, and
wait for answers before dependent questions. Record the agreed decisions and
obtain confirmation of the shared understanding before implementing the protocol.
This is a planning note, not a request to start that session or schedule an automation now.

Use a new study/output directory and the common pretrained checkpoint. Never
initialize it from a frozen comparison snapshot. The existing integrated
`freeze_pretrained_until_tokens` mechanism can provide a brief warmup without
starting from a different trajectory or resetting the added-group optimizer.

Decisions required before writing runnable configs:

1. Total linguistic-token budget, artifact/splits and effective optimizer batch.
2. No frozen warmup or one explicit warmup length shared under a stated policy.
3. Pass schedule and loss weights. A reasonable first proposal is to keep the
   frozen K=2/3 final-pass objective for comparability, but full-model training
   may benefit from a different objective; decide rather than inherit it silently.
4. Per-model tuning budget and candidate backbone/added-parameter LR pairs.
   The frozen optimal added LR is not automatically the unfrozen optimum.
5. Controls and reporting claims. A separately trained K=1 vanilla arm would
   test gains beyond ordinary backbone adaptation; same-checkpoint K=1 scores
   are useful but are not a substitute. Adding that arm needs a protocol decision.
6. Snapshot/feedback milestones, downstream evaluation budget and stopping/
   failure criteria. Match target
   sets, precision and artifacts; always record actual threshold-crossing counts.

Use the same evaluation module and independent prefill/decode parameters.
Downstream tasks retain actual-context K=4 prefill with ordinary feedback.
No new merger candidates or paper-replay implementation are proposed.

### 5. Report comparisons without hiding differences

Read arm IDs from study manifests and join results by snapshot identity and
actual token/optimizer counters. Produce NLL curves against tokens, estimated
FLOPs and training-only time, with end-to-end time and peak memory separately.
Report full-block BOS NLL and aligned-target BOS NLL separately from parallel K=4
NLL; equal target IDs still do not imply identical conditioning.

Check evaluation compatibility before combining runs: data/split/prefix,
precision, policy, target counts and checkpoint identity. Make changed settings
visible rather than pooling them. Final quality claims require appropriate
independent evaluation and more evidence than a single feedback block; preserve
the historical split-overlap caveats.

## Optional optimization, after measuring cost

Multi-block parallel validation batching remains useful but is not a prerequisite
for the next experiment. Add a real batch-size parameter distinct from
`max_blocks`/`eval_batches`, retaining a serial reference. Test batch sizes 1 and
larger on the same subset, including final partial batches, mixed sources and
MEM labels, with explicit numerical tolerances.

If feedback scoring is the bottleneck, reduce per-token host synchronization in
the shared scorer and benchmark the actual evaluator again. Preserve FP32 loss
accumulation semantics, bounded device memory, exact target counts and prompt
signal handling. Do not optimize by silently shortening the horizon or changing
the inference computation.

Keep this work behind the existing evaluation module's interface, shared by the
trainer and standalone commands. Avoid a new registry, worker framework, named
profile system or parallel study-specific evaluator unless a concrete need appears.

## Immediate next task

Confirm the bounded GPU execution budget and carry out the pre-training check,
using the production BOS evaluator for timing. Then run the existing LR sweep,
apply its selections and launch the main comparison when qualified. Use existing
simple ablations at early trained snapshots; defer token-zero evaluation and
expanded diagnostics. Hold the required grill session separately before
implementing or launching the unfrozen protocol.
