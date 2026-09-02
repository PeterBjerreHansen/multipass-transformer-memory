# Frozen-backbone comparison

This is a planned core candidate, not a locked campaign. All trained arms keep
the pretrained TinyMistral backbone frozen for the complete 20,021,248-token
trajectory. The initialized common checkpoint is the token-zero reference; it
is evaluated but is not a fake training arm.

## Effective batching and trajectory

Every arm uses `batch_size: 16` and `grad_accum_steps: 2`, consuming 32
sequences and 32,768 linguistic tokens per optimizer update. This is the
largest common physical batch qualified across all five arms on the target
A6000, so both optimizer batch and physical batch remain controlled. It also
amortizes much of the token-serial TBPTT launch overhead.

The TBPTT arm explicitly uses the reference attention backend because cached
token recurrence already falls back to that audited path; this avoids compiling
a one-token FlexAttention kernel at each process start. On the target A6000, a
true 1,024-token window-128 microbatch at physical batch 16 completed in 143.7
seconds with 3.25 GiB peak allocated memory. Full BPTT is excluded from the
active trajectory rather than being silently approximated.

On the target CUDA host, validate all five complete forward/backward paths
before starting the trajectories:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_comparison \
  --wire-only --wire-device cuda
```

After the qualifications and preflight pass, run the declared arms with the
same study executor. Its default auto-resume preserves the sampler, optimizer,
RNG, cumulative timer, and all scientific counters:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_comparison \
  --skip-wire
```

The retained snapshots land after optimizer updates 100, 153, 306, and 611:

| Unique input tokens | Optimizer updates | Purpose |
| ---: | ---: | --- |
| 3,276,800 | 100 | Recirculation-paper endpoint |
| 5,013,504 | 153 | Early continuation |
| 10,027,008 | 306 | Midpoint |
| 20,021,248 | 611 | Full frozen-backbone trajectory |

Training records are emitted every 327,680 tokens, or ten full optimizer
updates. `training_elapsed_seconds` is cumulative synchronized optimizer-update
time. It includes data transfer, forward, backward, gradient clipping, and the
optimizer step, but excludes validation, snapshot writing, and checkpoint I/O.
The counter is checkpointed, so it remains monotonic across automatic resumes.

## Bounded pilot gate

Do not start a full five-arm trajectory until the first shared endpoint has
been inspected. The pilot endpoint is 3,276,800 unique tokens (100 optimizer
updates). The runner can impose this bound without changing the authoritative
20M configs:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_comparison \
  --arm recirculation_tbptt_w128_20m \
  --skip-wire \
  --until-unique-tokens 3276800
```

For the current interrupted campaign, the four parallel arms already have
pilot snapshots at this endpoint. Resume only the TBPTT arm to the same
endpoint, then compare the common token-zero baseline, recirculation
multipass, recirculation TBPTT, and one representative Memory Attention arm.
The strided and multiscale arms remain useful completed trajectories but are
not required for the first go/no-go decision.

At the pilot endpoint, report held-out NLL under the applicable teacher-forced
views, convergence over validation passes 1--8 for multipass arms, and the
corresponding training-time/FLOP records. Run the downstream generation suite
separately, using feedback continuation only for its longer-generation tasks.
Extend only arms with a clear, semantically consistent signal to 10M tokens;
reserve the 20M endpoint for the final candidate.

## Evaluation and reporting

The TBPTT arm uses recurrent teacher-forced validation. The four parallel
multipass arms (recirculation, dense Memory Attention, Strided Memory Attention,
and Multiscale Memory Attention) retain whole-block pass-depth validation as a
diagnostic. Before comparing curves, evaluate every shared snapshot under both
applicable semantic views; K is not a parameter of the token-diagonal
recurrence.

Every parallel multipass wiring arm samples K=2 with probability 0.9 and K=3
with probability 0.1, matching the continual-training schedule. Its NTP loss is
final-pass-only: `[0, 1]` for K=2 and `[0, 0, 1]` for K=3. The zero first-pass
weight is intentional in Phase A because the pretrained backbone is frozen and
only the added feedback mechanism is being wired. The TBPTT arm remains K=1 and
has no multipass loss weights because its recurrence is token-diagonal rather
than a pass-depth axis. K=2/K=3 applies to prefill/training and validation
diagnostics; generation is evaluated separately with the feedback continuation
mechanism.

The `strided_memory_attention` arm is learned Memory Attention with periodic
stride-32 writes. It is distinct from the parameter-free `strided_attention`
control, which has no Phase-A wiring parameters and is therefore not an arm in
this study. `multiscale_memory_attention` retains a dense recent window and a
strided sparse older window.

Use each arm's single trajectory to report held-out NLL against:

1. unique linguistic training tokens;
2. optimizer updates;
3. cumulative `training_elapsed_seconds`;
4. cumulative estimated dominant training FLOPs.

Also report interval tokens/s, peak VRAM from the CUDA qualification, and total
end-to-end wall time separately. Do not include validation or checkpoint time
in the training-time curve.

Generate the study-specific FLOP report with:

```bash
make estimate-flops-frozen-backbone
```

The resulting `results/training_flops.json` is derived directly from the five
authoritative arm configs. It counts dominant matrix operations, uses the
conventional forward-plus-backward multiplier, and counts activation-checkpoint
recomputation. It excludes elementwise operations, optimizer arithmetic, and
does not discount missing frozen-parameter gradients. Treat it as a transparent
architecture-normalized estimate, not a hardware counter; measured accelerator
time is the practical-efficiency result.

## Promotion gate

Promote this directory to `benchmarks/core/` and set `status: locked` only after
the forward-mode, CUDA-memory, TBPTT-window, and learning-rate
qualifications are complete.
