# Frozen-backbone comparison

This is a planned exploratory campaign, not a locked study. The four default
trained arms keep the pretrained TinyMistral backbone frozen for the complete
100,007,936-token trajectory on 2,048-token blocks from
`data/dolmino/gpu_2048`. The initialized common checkpoint is the token-zero
reference; it is evaluated but is not a fake training arm.

The 100M campaign uses the per-mechanism added-module learning rates selected by
the 2,048-token qualification. Until those results exist, the checked-in
`3e-4` values are provisional. The LR schedule remains constant because these
curves measure controller adaptation under a fixed step size.

## Effective batching and trajectory

Every default arm uses `batch_size: 8` and `grad_accum_steps: 4`, consuming 32
sequences and 65,536 linguistic tokens per nominal optimizer update. If a new
hardware profile cannot fit batch 8, an explicit fallback may use 4x8, 2x16, or
1x32 while preserving the nominal optimizer batch.

The token-diagonal TBPTT policy is qualified separately by
`forward_policy_qualification/` and is not part of this default study.

On the target CUDA host, validate all four default forward/backward paths before
starting the trajectories:

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

The retained snapshots remain defined by linguistic-token milestones. Because
the block length and microbatch policy changed, the corresponding optimizer
update counts differ; reports should use the checkpointed actual token and
update counters rather than assuming the old update numbers:

| Requested input tokens | Optimizer updates reached | Purpose |
| ---: | ---: | --- |
| 3,276,800 | 50 | Early snapshot |
| 5,013,504 | 77 | Early continuation |
| 10,027,008 | 153 | Midpoint |
| 20,021,248 | 306 | Early long-run snapshot |
| 50,003,968 | 763 | Midpoint snapshot |
| 100,007,936 | 1526 | Full frozen-backbone trajectory |

Training records are emitted every 327,680 tokens, or ten full optimizer
updates. `training_elapsed_seconds` is cumulative synchronized optimizer-update
time. It includes data transfer, forward, backward, gradient clipping, and the
optimizer step, but excludes validation, snapshot writing, and checkpoint I/O.
The counter is checkpointed, so it remains monotonic across automatic resumes.

## Evaluation and reporting

The four default parallel multipass arms (recirculation, dense Memory
Attention, Strided Memory Attention, and Multiscale Memory Attention) retain
whole-block K=4 validation as the headline training-distribution curve, with
K=1 through K=4 values retained as diagnostics. K is a pass-depth axis, not a
longer-context or generation setting. Online teacher-forced feedback remains
an optional separate diagnostic; downstream generation is evaluated separately.

Every parallel multipass wiring arm samples K=2 with probability 0.9 and K=3
with probability 0.1, matching the continual-training schedule. Its NTP loss is
final-pass-only: `[0, 1]` for K=2 and `[0, 0, 1]` for K=3. The zero first-pass
weight is intentional in Phase A because the pretrained backbone is frozen and
only the added feedback mechanism is being wired. K=2/K=3 applies to
prefill/training and validation diagnostics; generation is evaluated separately
with the feedback continuation mechanism. TBPTT remains a separate training
policy with K=1 and no multipass loss weights because its recurrence is
token-diagonal rather than a pass-depth axis.

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

The `gpu_2048` artifact contains roughly 2M validation tokens. Routine training
checks continue to use a small fixed validation sample; run the complete
validation stream only at selected milestones when its cost is justified.

Generate the study-specific FLOP report with:

```bash
make estimate-flops-frozen-backbone
```

The resulting `results/training_flops.json` is derived directly from the four
authoritative default arm configs. It counts dominant matrix operations, uses the
conventional forward-plus-backward multiplier, and counts activation-checkpoint
recomputation. It excludes elementwise operations, optimizer arithmetic, and
does not discount missing frozen-parameter gradients. Treat it as a transparent
architecture-normalized estimate, not a hardware counter; measured accelerator
time is the practical-efficiency result.

## Promotion gate

Promote this directory to `benchmarks/core/` and set `status: locked` only after
the forward-mode, CUDA-memory, and learning-rate qualifications for the four
default arms are complete. The separate TBPTT policy is promoted only if the
paper explicitly includes that training-policy comparison.
