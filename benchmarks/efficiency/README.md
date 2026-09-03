# Efficiency benchmarks

These suites measure implementation cost and hardware feasibility. They do not
rank model quality and are not scientific training arms.

The 1024-token paper-policy qualification, suite and Makefile targets have been
deleted, including the temporary historical copies; paper replay/BPTT
execution is removed. The FLOP script now defaults to `suites/training.yaml`.

## General engineering suites

The remaining active suites measure ordinary training, batch scaling, context
scaling, and precision behavior across CUDA or MPS:

```text
suites/training.yaml
suites/batch_scaling.yaml
suites/context_scaling.yaml
suites/precision_cuda.yaml
suites/precision_mps.yaml
```

These are engineering grids, not authoritative arm configurations. Their
recirculation cases use the older shifted middle-layer model, not the two
late-memory mergers in the active frozen study. Context-scaling suites may
include shorter inputs without reviving a 1024-token scientific study. Use the explicit Makefile
targets to run them. Successful rows report linguistic and physical-position
throughput, optimizer-step timing, projected hours, precision, and available
memory telemetry.

The runner uses deterministic synthetic token IDs so data loading does not
contaminate measurements. For sequence length `T`:

```text
microbatch_tokens = batch_size * T
optimizer_batch_tokens = batch_size * grad_accum_steps * T
```

Small retained JSON summaries belong under `results/`; generated measurements
may remain local. The separate historical efficiency directory has been removed.

Feedback-evaluation timing is recorded separately in the
[A6000 report](../development/inference_efficiency/README.md). Its expensive
combined exact-vs-feedback diagnostic must not be labelled feedback-only cost.
Completed cleanup and remaining evaluation additions are tracked in
[docs/CLEANUP_STATUS.md](../../docs/CLEANUP_STATUS.md).
