# Staged implementation plan

Updated 2026-09-05. This is the implementation sequence for the agreed
[research plan](RESEARCH_PLAN.md). Current behavior remains defined by the
architecture, training and evaluation contracts until a stage below is complete.
This plan does not authorize paid cloud runs.

Implementation status: Stages 0–6 are implemented in the working tree. The
stride ablation was fixed before results at two sites with physical strides
`C = 8, 16, 32, 64`, memory capacity 32, and corresponding effective spans
of 256, 512, 1,024, and 2,048 physical tokens. Stage 7 remains the next stage;
its target-GPU checks have not been run.

## Design constraints

- Preserve the explicit distinction between `exact_k_pass`, `live_feedback` and
  `standard_k1` execution.
- Keep the existing `run_memory_decoder(..., after_attention=...)` seam for
  architecture-specific reads. Do not force native mergers through a common
  mathematical implementation.
- Put shared execution and diagnostic policy behind one interface instead of
  duplicating it in each command or variant.
- Keep diagnostics separate from routine validation and training control flow.
- Derive comparison settings from runnable configs and study manifests; do not
  maintain a second hidden source of experiment defaults.
- Do not add dummy parameters solely to satisfy parameter matching.
- Preserve checkpoint provenance and explicit failure on incompatible configs.

## Stage 0: freeze the plan and inventory the delta

**Work**

1. Record the research contract and canonical vocabulary.
2. Compare current configs with the required one-site and two-site matrix.
3. Produce a parameter/FLOP inventory for every current reader and merger.
4. Identify serialized names and scripts affected by the feedback terminology
   change and legacy Recirculation archival.

**Completion check**

- The documentation map links the research and implementation plans.
- The inventory identifies the width/rank changes needed to place every arm
  within the agreed comparison budget.
- No runnable config is changed before the inventory is reviewed.

## Stage 1: make execution semantics explicit

**Work**

1. Change `MultiPassVariant.forward()` and `.generate()` so ambiguous generic
   calls fail with an explanation of the explicit scoring and decoding paths.
2. Rename generic recurrent-continuation modules, commands, result fields and
   documentation to Live Feedback terminology.
3. Use `exact_k_pass`, `live_feedback` and `standard_k1` consistently in new
   result schemas. Keep compatibility translation only at serialized input
   seams where existing artifacts require it.
4. Remove the old middle-layer `RecirculationVariant` from active study/factory
   choices and mark its retained implementation and records as legacy. Do not
   rewrite historical outputs.

**Completion check**

- Generic `forward()` and `generate()` cannot silently evaluate the vanilla
  backbone for a feedback variant.
- The LM-evaluation adapter and standalone commands still reach the intended
  explicit execution mode.
- Naming tests reject ambiguous new `recurrent` labels for generic feedback
  execution while permitting mechanism-specific recurrent names.

## Stage 2: make stride origin explicit and test it

**Work**

1. Preserve the existing Strided Memory Attention rule: zero-based physical
   sequence position `t` writes when `(t + 1) % C == 0`.
2. Make full-sequence, cached prefill and single-token append paths use that same
   physical-position origin for every configured stride.
3. Treat the BOS-only evaluator's synthetic BOS as a real physical context
   position under ordinary Live Feedback. Record that this shifts the stride
   phase of later data tokens relative to an unprefixed block.
4. Remove or qualify metadata that implies the synthetic BOS leaves Strided
   Memory Attention cadence unchanged. If a data-phase-aligned diagnostic is
   later useful, expose it as a separate evaluation policy rather than changing
   ordinary Live Feedback semantics.
5. Verify dense behavior remains unchanged.

**Completion check**

- Parameterized tests cover several strides. Without a prefix, writes occur at
  data indices `C-1, 2C-1, ...`; after one synthetic BOS, they occur at data
  indices `C-2, 2C-2, ...` for `C > 1`.
- Full-sequence and cached paths agree on write triggers, retained positions and
  logits.
- Evaluation provenance states the stride origin and whether a synthetic prefix
  changes the data-token phase.
- Existing strict-past and position-zero invariants continue to pass.

## Stage 3: add the No-memory Adapter and budget controls

**Work**

1. Implement a No-memory Adapter through the existing post-self-attention,
   pre-MLP decoder seam.
2. Activate it only on feedback passes. It may consume the current residual
   state but no earlier-pass or earlier-token Feedback Record.
3. Support explicit one-site `[3]` and two-site `[3, 7]` configurations.
4. Make only the necessary controller, projection, adapter-rank or memory-reader
   dimensions configurable for real parameter matching.
5. Generate a checked budget table from instantiated models and the FLOP
   estimator. Match added parameters within approximately 10 percent inside
   each site-count group and keep estimated FLOPs close enough for overlapping
   compute curves.
6. Prefer the smallest width adjustment that preserves each native mechanism.
   If exact matching would require a qualitatively different architecture,
   retain the nearest reasonable configuration and disclose the residual gap.

**Completion check**

- With fixed inputs, the adapter's output is independent of prior-pass states.
- Pass 1 remains the ordinary backbone path.
- One-site and two-site parameter/FLOP tables are reproducible from configs.
- All architecture-added parameters receive gradients in their intended native
  initialization schedule.

## Stage 4: deepen the intervention diagnostic

The evaluation module owns this behavior; the command remains a thin adapter
for loading inputs and writing results.

**Work**

1. Add a shared feedback-transition interface that can execute a real, zero,
   mismatched or bypassed feedback condition without evaluator-specific access
   to merger internals.
2. Define true bypass as the same decoder execution with the memory injection
   omitted, not as a zero-valued source passed through the merger.
3. Make the target transition configurable. For transition into pass `p`, build
   passes `1..p-1` normally, apply one intervention, and score pass `p`.
4. For mismatched memory at depth `p`, obtain the donor from a different block
   processed normally through pass `p-1`.
5. Run transitions into passes 2, 3 and 4 without adding them to routine
   validation.
6. Continue reporting NLL, target counts and hidden deltas per source and
   condition.

**Completion check**

- Adaptive recirculation with zero memory can differ from true bypass, and a
  focused test demonstrates that the evaluator records both conditions.
- Every active arm supports all four conditions at each transition through K=4.
- Donor depth, target transition, source block and precision are recorded in the
  result provenance.

## Stage 5: extend exact-versus-Live-Feedback fidelity

**Work**

1. Extend the existing continuation diagnostic rather than ordinary feedback
   NLL validation.
2. Always report teacher-forced, per-offset and cumulative-horizon:

   - exact, Live Feedback and standard-K1 NLL;
   - `KL(exact || live_feedback)` in FP32;
   - top-1 agreement;
   - hidden-state RMS and cosine drift; and
   - retained K=4 NLL improvement when its denominator is positive.

3. Keep free-running stability optional because it has a different cost and
   stochastic contract. Record decoding parameters when enabled.
4. Preserve same-prefill initialization so divergence begins only after the
   shared exact prefill state.

**Completion check**

- The first continuation prediction agrees where exact and Live Feedback are
  defined to share state.
- Metrics have explicit target counts at every offset and horizon.
- Identical logits produce zero KL and complete top-1 agreement in reference
  tests.
- No binary fidelity threshold is embedded in evaluation or training control.

## Stage 6: materialize the wiring studies

**Work**

1. Replace the current unequal-site frozen study with two internally matched
   study groups:

   | Group | Sites | Arms |
   | --- | --- | --- |
   | One-site dense | `[3]` | No-memory, projected residual, Recirculation-inspired, dense attention |
   | Two-site dense | `[3, 7]` | No-memory, projected residual, Recirculation-inspired, dense attention |

2. Preserve the agreed 100M-token frozen-backbone protocol, native
   initialization, per-microbatch 90/10 K=2/K=3 sampling and final-pass loss.
3. Apply the same predefined LR grid to every newly parameterized arm. Do not
   search additional architecture choices after observing full-run results.
4. Add one-site and two-site Strided Memory Attention only after the dense groups
   pass preflight; add Dense-and-strided Memory Attention afterward.
5. Preserve the planned stride-length sub-study. Choose its site-count group
   before looking at stride results, then vary only `memory_write_stride` while
   holding memory capacity, parameterization, data and training schedule fixed.
6. Keep preflight, qualification and scientific trajectories in separate output
   directories. A qualification checkpoint is not the beginning of a 100M run.

**Completion check**

- Study verification proves common checkpoint, data, token order, target set,
  precision, snapshots and site count within each group.
- Generated manifests record parameters, estimated FLOPs, K sampling and native
  initialization for every arm.
- Stride-ablation manifests also record physical write counts and effective
  memory span for every tested stride.
- Config-backed documentation contains no obsolete five-arm or unequal-site
  frozen protocol.

## Stage 7: execute correctness and target-GPU preflight

**Work**

1. Run the full test suite.
2. Run finite-loss and gradient smoke tests for every arm and site count.
3. Compare cached and full-sequence exact K-pass outputs.
4. Run small real/zero/mismatch/bypass and exact-versus-live diagnostics.
5. Benchmark complete optimizer updates, validation, snapshots and resume on the
   target GPU.
6. Verify interruption and exact recovery before any long trajectory.

**Completion check**

- No arm silently falls back to vanilla execution.
- All trainable groups update and all frozen groups remain unchanged.
- Diagnostics contain finite values, exact counts and complete provenance.
- Measured memory fits with the selected batch/accumulation configuration.
- A projected time and cost exists for qualification and 100M runs.

## Stage 8: qualify and run the frozen wiring experiment

**Work**

1. Run equal-budget LR qualification for the dense one-site and two-site groups.
2. Lock one LR per arm using held-out NLL and basic numerical stability.
3. Launch fresh 100M trajectories for the dense groups.
4. Inspect normal validation first, then run the separate intervention and
   fidelity diagnostics at selected snapshots.
5. Run the strided and dense-and-strided extensions in the agreed order.
6. Produce curves against unique tokens, estimated FLOPs and measured time.
7. Replicate only decisive finalists after the pipeline and selection policy are
   stable.

**Completion check**

- Every plotted point resolves to a checkpoint and study manifest.
- One-site and two-site comparisons are never pooled as if site count were held
  constant.
- Parallel K=4, Live Feedback and standard K=1 remain separately labelled.
- Intervention results are diagnostics, not routine stopping criteria.

## Stage 9: specify and preflight the full-model scaling study

**Work**

1. Create a new study and output root. Initialize every arm from the common
   pretrained checkpoint, never from wiring weights.
2. Select and lock dense attention, the strongest sparse/multiresolution
   attention arm, the strongest fixed-route arm, No-memory Adapter and vanilla.
3. Unfreeze the entire backbone and all added parameters from token zero.
4. Retain the locked 90/10 K schedule and final-pass loss unless a separately
   approved objective experiment changes them.
5. Materialize one common approximately one-billion-token corpus. Feedback arms
   traverse it once; vanilla deterministically cycles the same blocks until its
   cumulative estimated FLOPs match.
6. Record unique tokens and token presentations independently.
7. Re-estimate full-model training FLOPs and benchmark measured throughput; the
   earlier universal frozen-backward estimate is not sufficient.
8. Verify snapshots and exact resume, then project per-arm runtime and cost.
   Stay single-device if acceptable; add distributed execution only if the
   measured projection requires it.

**Completion check**

- Finalists and selection evidence are recorded before long configs are locked.
- Vanilla receives no additional unique corpus data during its compute-matched
  tail.
- Equal-data and equal-compute comparisons both have overlapping curve support.
- Reporting language says small parameter overhead, not trainable-parameter
  efficiency, for the fully unfrozen study.

## Stage 10: run and report the scaling study

**Work**

1. Launch fresh full-model trajectories only after Stage 9 preflight.
2. Evaluate the same standard-K1, parallel-K4 and Live-Feedback estimands at
   aligned checkpoints.
3. Report quality against unique tokens, token presentations, estimated FLOPs,
   training-only time and end-to-end time.
4. Report total, added and optimized parameters, optimizer-state memory and peak
   device memory.
5. Treat Live Feedback failure despite parallel K=4 gains as failure of the
   training surrogate for the intended deployed model.
6. Keep any prefix-mixed or Live-Feedback-aware training follow-up in a new,
   explicitly labelled study.

**Completion check**

- Equal-data and equal-compute claims use the corresponding checkpoints rather
  than endpoint-only comparisons.
- Results from different target sets, precisions or execution modes are not
  pooled.
- Scientific wording remains within the limits in the research plan.

## Deferred work

- Faithful FBT, T²MLR or published Recirculation reproduction.
- A looped-transformer literature baseline.
- Synthetic state-tracking tasks.
- Intermediate-pass auxiliary losses or a different K mixture.
- Live-Feedback-aware training and prefix mixing.
- Broad injection-layer searches.
- Multi-device training before measured single-device need.

## Immediate next stage

Run Stage 7's correctness and target-GPU preflight. Do not launch qualification
or paid scientific runs until its completion checks pass.
