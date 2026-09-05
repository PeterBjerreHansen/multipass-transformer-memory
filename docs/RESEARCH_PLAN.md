# Distributed Feedback Memory research plan

Updated 2026-09-05. This document records the scientific motivation, comparison
contract and claim limits agreed for the MVP and its longer scaling experiment.
It does not describe current implementation behavior or authorize paid runs.

## Thesis

Autoregressive transformers can carry renewed-depth computation forward through
Temporal Feedback. A fixed-route mechanism exposes one predetermined preceding
record to the current token. Memory Attention instead retains multiple records
and lets the current representation retrieve a content-dependent mixture.

The project asks whether representative Memory Attention architectures are more
effective than representative Fixed-route State Injection architectures when the
backbone, data, injection count, added-parameter budget and training compute are
held as close as practical.

The intended title is **Distributed Feedback Memory for Multi-Pass Transformers**.
The longer description is **Beyond the Recurrent Bottleneck: Attention Over
Feedback Memory in Multi-Pass Transformers**. "Recurrent bottleneck" refers to
the fixed-route read interface, not to the transformer's entire history or to a
claim that attention is non-recurrent in the broad dynamical-systems sense.

## Model and training semantics

Let `d[t]` be the current residual state and `m[j]` a retained Feedback Record.
A representative fixed-route reader has the form

```text
r[t] = G(d[t], m[t-1])
```

where the source record is determined before inspecting the current content. A
representative attention reader has the form

```text
r[t] = sum(j in S[t]) a(d[t], m[j]) V(m[j])
```

where `S[t]` is a strict-past set and the normalized weights depend on the
current query and stored keys. The comparison concerns representative complete
architectures. Their native normalization, projection, gating and merger rules
are allowed to differ; the experiment does not claim to identify record access
as the only mathematical cause of an observed difference.

The deployed object of interest is Live Feedback during autoregressive inference.
Parallel Multi-pass Training is its Jacobi-style training surrogate. Parallel
K-pass evaluation estimates teacher-forced refinement quality, but it is not a
substitute for Live Feedback evaluation. Exact K-pass Continuation is a reference
used to measure how the one-stream policy diverges after the common prefill.

## Contribution and claim limits

The primary contribution is a controlled empirical comparison of feedback
architecture classes in a pretrained-transformer retrofit. A longer experiment
tests whether the observed behavior persists with more data and full-model
training.

The project does not claim:

- state-of-the-art language-model quality;
- that attention over feedback is itself unprecedented;
- that the selected architectures are exhaustive or globally optimized;
- that attention universally dominates recurrent computation;
- that the Recirculation-inspired Merger reproduces the published Recirculation
  execution or training policy; or
- that parallel K-pass gains establish successful Live Feedback by themselves.

[Feedback Transformer](https://arxiv.org/abs/2002.09402) and
[TransformerFAM](https://arxiv.org/abs/2404.09173) are direct precedents for
attention over feedback state. [Full Bandwidth Transformer](https://arxiv.org/abs/2608.08888)
is the closest training and inference reference for Jacobi-style temporal
feedback. [Recirculation](https://arxiv.org/abs/2608.17981) and
[T²MLR](https://arxiv.org/abs/2607.15178) motivate fixed-route comparators, but
their complete published methods are not MVP arms. The active adaptive comparator
borrows the Recirculation mixing rule and applies it inside this project's common
late-writer retrofit.

## Experiment 1: frozen-backbone wiring

### Purpose

Determine whether the choice of feedback architecture matters when only the
added wiring is trained. This experiment is a short retrofit study, not evidence
about full-model pretraining or frontier performance.

### Shared contract

- Start every run from the same pretrained TinyMistral checkpoint.
- Freeze the pretrained backbone.
- Use independently trained writers with the same architecture and the same
  final-normalized-state source definition.
- Enforce strict-past feedback.
- Use the same injection locations within each site-count group.
- Match added trainable parameters within approximately 10 percent and keep
  estimated training FLOPs as close as practical.
- Use native initialization and the same predefined learning-rate qualification
  grid for every trained arm.
- Sample K per microbatch with 90 percent K=2 and 10 percent K=3.
- Optimize final-pass loss only.
- Use the same data, token order, snapshots, scoring targets and precision.
- Run one seed while establishing correctness, then replicate decisive finalists
  after the pipeline is stable.

### Arm sequence

The first group uses one injection site at layer `[3]`. The second uses two sites
at layers `[3, 7]`. Each group contains:

1. No-memory Adapter;
2. projected-residual Fixed-route State Injection;
3. Recirculation-inspired Fixed-route State Injection; and
4. Dense Memory Attention.

One-site and two-site groups are matched internally. Their cross-group difference
is a practical architecture ablation, not a pure estimate of injection-count
causality.

After the dense groups are healthy, run one-site and two-site Strided Memory
Attention, followed by Dense-and-strided Memory Attention. These layouts test
whether access to regularly spaced older records remains competitive when recent
records are omitted or combined with dense memory.

Preserve a separate stride-length ablation for Strided Memory Attention. Hold the
chosen site count, memory-window capacity, data, training schedule and parameter
configuration fixed while varying the physical-position write stride `C`. Report
the resulting write count, effective memory span and measured compute for each
stride. This ablation tests memory spacing; it is not folded into the primary
dense-versus-fixed-route comparison.

The stride origin is the actual physical input sequence: zero-based position `t`
writes when `(t + 1) % C == 0`. A synthetic BOS prepended by a diagnostic is an
additional physical position and therefore shifts the stride phase of subsequent
data tokens. That diagnostic must report the shift rather than silently presenting
its memory cadence as identical to an unprefixed training block.

### Evidence

Routine results report standard K=1, parallel K=4 and contextual-prefill Live
Feedback quality. Separate diagnostics report:

- Exact K-pass versus Live Feedback NLL by continuation horizon;
- token-distribution divergence and top-1 agreement;
- hidden-state RMS and cosine drift;
- the fraction of the K=4 NLL improvement retained by Live Feedback; and
- real, zero, mismatched and bypassed memory at each transition through K=4.

There is no postulated binary fidelity threshold. Results are reported as
continuous measurements. Qualitative claims must follow the observed evidence.

## Experiment 2: full-model scaling

### Purpose

Test whether the wiring result survives a fresh, longer run when the entire model
is trainable. This experiment studies data and compute scaling with small added
parameter overhead. It does not claim fewer optimized parameters than vanilla.

### Arms

Every arm restarts from the common pretrained checkpoint rather than a wiring
snapshot. The planned arms are:

1. Dense Memory Attention;
2. the strongest sparse or dense-and-strided attention finalist;
3. the strongest fixed-route finalist;
4. the No-memory Adapter; and
5. vanilla continued pretraining.

Wiring results select the finalists before long-run configurations are locked.
All backbone and added parameters are trainable from token zero.

### Data and compute comparison

All arms use the same fixed set of approximately one billion unique training
tokens. Feedback arms traverse that set once under the locked K=2/K=3 schedule.
Vanilla cycles through the same blocks until it reaches the feedback arms'
cumulative training compute, expected to require roughly two token presentations
per unique token. It does not receive additional unique training data.

Reports distinguish:

- unique corpus tokens;
- total token presentations;
- estimated training FLOPs;
- measured training-only and end-to-end time;
- peak memory;
- total, added and optimized parameter counts; and
- optimizer-state memory.

The frozen experiment may support a low-trainable-parameter retrofit claim. The
fully unfrozen experiment may support data- or compute-efficiency claims and a
small-parameter-overhead claim, but not trainable-parameter efficiency.

## Evaluation hierarchy

Parallel K=4 and Live Feedback are separate estimands. Parallel K=4 diagnoses
the training surrogate under teacher forcing. Live Feedback evaluates the
intended deployed temporal system. Standard K=1 records ordinary behavior from
the same checkpoint.

If parallel K=4 improves while Live Feedback is unstable or loses the improvement,
the MVP conclusion is that the training surrogate failed for the intended model.
Live-feedback-aware training or prefix mixing would be a separately labelled
follow-up, not an invisible rescue of the original condition.

## Deferred scope

The MVP does not require faithful FBT, T²MLR or published Recirculation
reproduction, synthetic state-tracking tasks, intermediate-pass auxiliary losses,
live-feedback-aware training, broad injection-site searches or claims of
architectural optimality. A published-method or looped-transformer baseline may
be added later if it materially sharpens the final paper comparison.
