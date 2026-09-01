# Frozen-backbone comparison

This is a planned core candidate, not a locked campaign. All trained arms keep
the pretrained TinyMistral backbone frozen for the complete 20,021,248-token
trajectory. The initialized common checkpoint is the token-zero reference; it
is evaluated but is not a fake training arm.

## Fixed batching and trajectory

Every arm deliberately uses `batch_size: 1` and `grad_accum_steps: 32`. This is
a study invariant, not a target-GPU tuning suggestion. Each optimizer update
therefore consumes 32 sequences and 32,768 linguistic tokens. Using the same
physical microbatch also makes the measured A6000 time comparison a controlled
implementation comparison, although it may leave throughput available to the
parallel methods unused.

Do not increase the physical batch for one arm or silently replace full BPTT
with TBPTT. If full BPTT does not fit at microbatch one, stop and revise the
protocol explicitly.

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

## Evaluation and reporting

The BPTT arm uses recurrent teacher-forced validation. The four parallel
multipass arms (recirculation, dense Memory Attention, Strided Memory Attention,
and Multiscale Memory Attention) retain whole-block pass-depth validation as a
diagnostic. Before comparing curves, evaluate every shared snapshot under both
applicable semantic views; K is not a parameter of the token-diagonal
recurrence.

Every parallel multipass wiring arm samples K=2 with probability 0.9 and K=3
with probability 0.1, matching the continual-training schedule. Its NTP loss is
final-pass-only: `[0, 1]` for K=2 and `[0, 0, 1]` for K=3. The zero first-pass
weight is intentional in Phase A because the pretrained backbone is frozen and
only the added feedback mechanism is being wired. The BPTT arm remains K=1 and
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
the forward-mode, CUDA-memory, truncation-window, and learning-rate
qualifications are complete.
