# Development plan

Updated 2026-09-03. This plan follows the completed cleanup. It proposes the next
implementation and experiment tasks; it does not authorize paid runs, change
study defaults, or select an unfrozen protocol on the user's behalf.

## Current position

The code supports the intended comparison, but the restructured five-arm study
is still planned, not GPU-qualified or scientifically concluded.

- Active arms: two late-memory recurrent mergers (`projected_residual` and
  adaptive `recirculation`), plus dense, strided and multiscale Memory Attention.
- All use preceding-token/previous-pass memory. The recurrent pair shares the
  writer rule and read location. Paper replay/BPTT and the 1024 studies are gone.
- The frozen configs use 2048-token blocks, K=2/3 final-pass training, K=4
  routine validation, and a four-rate qualification grid for each model.
  Long-run rates are still provisional.
- Shared evaluators record precision, scored targets, subset and weight identity.
  BOS-only feedback NLL is available standalone and after selected snapshots;
  it reports full and aligned target sets. Its schedule is disabled by default.
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
| 1 | Initial-state and K=4 memory-use diagnostics | One shared evaluator with tests, plus a small comparable time-zero report for all five arms |
| 2 | Bounded target-GPU qualification | Actual five-arm training/validation fit, directly measured BOS NLL time, and an interruption/resume check |
| 3 | Frozen per-model LR selection | The existing 20-arm qualification, a documented selection for each model, then resolved long-run configs |
| 4 | Fresh unfrozen protocol and LR qualification | A separate study and per-model backbone/added-parameter sweep, starting from pretrained weights |
| 5 | Comparison reporting | Snapshot-aligned frozen/unfrozen reports with explicit target and computation differences |

Specify the unfrozen protocol while the short frozen qualification is underway.
Do not make it wait for completion of every 100M frozen trajectory. Its weights,
optimizer and learning-rate choices must nevertheless be independent.

### 1. Measure whether the memory is used

The current intervention evaluator tests only the K=1-to-K=2 transition. Extend
that module to an explicit pass depth and intervention transition; use a shared
K=1–3 prefix when testing the final K=3-to-K=4 transition. Generate mismatch
donors at the same depth and preserve strict-past routing.

Compare real, zeroed and mismatched memory, and distinguish those interventions
from bypassing the merger entirely. Zero source is not automatically a
no-feedback control: adaptive recirculation can still rescale the destination.
Keep donor selection, target masks and precision explicit in the result.

Run the same small check at initialization and selected trained snapshots. Record
K=1–4 NLL and relevant writer/merger scales. The projected residual starts as a
no-op; adaptive recirculation does not. Report initial absolute loss and subsequent
change, not only improvement from each arm's different starting loss.

Completion checks: real-memory K=4 agrees with ordinary K=4 validation; a no-op
path behaves as expected; all conditions score identical targets; mismatches
come from another block; tests exercise both recurrent mergers and the attention
arms. Keep this a small diagnostic, not another architecture search or a demand
that every model pass a hand-chosen quality threshold before experimentation.

### 2. Qualify the actual execution path

On the intended GPU, use the resolved arm configs and preserve the effective
optimizer batch when adjusting microbatch size. Measure:

- representative frozen and, once specified, unfrozen forward/backward steps;
- the normal 64-block K=4 validation check;
- the actual `evaluate_feedback_nll` computation on one complete 2048-token
  block, including scoring and host transfers, with FP32 and BF16 where supported;
- a selected snapshot/check/report cycle, interrupted during feedback and then
  resumed, checking that training does not advance early or duplicate reports.

Keep synthetic cache curves and combined exact-vs-feedback diagnostics separately
labelled. The timing script now identifies arms rather than collapsing the two
recurrent mergers by variant name, but still needs an actual BOS-evaluator timing
case. Prefer calling the production evaluator over copying its token loop.

Completion checks: finite outputs, expected target counts, acceptable memory use,
measured cost per selected checkpoint, and confirmed recovery. Then select an
explicit common feedback schedule and block prefix for all arms. Proposed first
use: one block at a small number of existing durable milestones, not every eval.
No full-split feedback campaign is implied.

### 3. Finish frozen qualification and run the comparison

Use the existing five-model, four-rate, 5,013,504-token qualification. Select each
model's added-parameter LR using K=4 validation NLL and basic stability evidence,
recording the equal tuning budget. Do not reuse old middle-layer/1024 selections
as qualification of the new recurrent arms.

Update the five long-run configs with the selected rates and chosen feedback
milestones. Keep one trajectory per arm, durable snapshots, the same data and
routine target set. If LR candidates remain indistinguishable, decide whether a
small finalist extension is worth its cost; do not silently expand one model's
search budget. An individual 5M qualification trajectory is not automatically
the first segment of the final 100M run.

### 4. Define the fresh unfrozen experiment

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
6. Snapshot/feedback milestones and downstream evaluation budget. Match target
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

Implement package 1 and the production-evaluator timing case from package 2,
with tests and documentation. Then request/confirm the bounded GPU execution
budget. In parallel with planning, resolve the six unfrozen protocol choices
above. Do not launch long trajectories merely because the cleanup is complete.
