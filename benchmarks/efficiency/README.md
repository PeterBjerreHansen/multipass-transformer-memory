# Efficiency benchmarks

These suites measure implementation cost and hardware feasibility. They do not
rank model quality and are not scientific training arms.

## Paper-contract qualification

`suites/forward_modes.yaml` is the only architecture-specific active suite. It
uses the paper contract's 1,024-token blocks and compares:

- vanilla one-pass training;
- whole-block K=2 Recirculation;
- token-diagonal Recirculation with TBPTT windows 128, 256, and 512;
- dense Memory Attention at K=2.

Run it on the target GPU before locking the scientific studies:

```bash
make efficiency-cuda-forward-modes
make estimate-flops-forward-modes
```

All cases use physical batch 16 and accumulation 2, the largest common policy
qualified for the frozen-backbone study on the target A6000. TBPTT uses the
reference cached-attention backend. `batch_size * grad_accum_steps` remains 32;
a different effective optimizer batch or learning rate is a protocol change,
not an invisible hardware adjustment.

Full BPTT is the paper-faithful gradient reference but is not in the active CUDA
suite because the serial implementation is operationally infeasible at length
1,024. A finite TBPTT window preserves the forward KV trajectory but truncates
the gradient path, so it remains an explicit experimental axis.

For token-diagonal recurrence, `passes: 1` denotes one recurrent forward policy;
it does not mean one ordinary backbone pass per token. Use measured runtime or
the architecture-aware FLOP estimator for compute claims.

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

They retain their original 2,048-token engineering grid because they test the
implementation rather than define the paper protocol. Use the explicit Makefile
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
may remain local. Suites tied to the superseded Stage-5 architecture set are in
`../historical/efficiency/`.
