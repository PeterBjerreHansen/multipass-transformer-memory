# Frozen-backbone comparison

This exploratory study compares matched feedback wiring while the pretrained
backbone stays frozen. It is not a locked paper study; each arm starts fresh
from the common pretrained checkpoint. Earlier GPU trajectories were stopped
and discarded after the tokenizer-padding audit. Regenerate the clean artifact,
rerun qualification, and then launch these trajectories from token zero.

## Arms

| Group | Memory mechanism | Reader layers | Added parameters |
| --- | --- | --- | --- |
| One-site | No-memory Adapter | `[3]` | 4,197,376 |
| One-site | Projected-residual fixed route | `[3]` | 4,197,376 |
| One-site | Recirculation-inspired fixed route | `[3]` | 4,195,000 |
| One-site | Dense Memory Attention | `[3]` | 4,196,352 |
| Two-site | No-memory Adapter | `[3, 7]` | 7,346,176 |
| Two-site | Projected-residual fixed route | `[3, 7]` | 7,346,176 |
| Two-site | Recirculation-inspired fixed route | `[3, 7]` | 7,341,424 |
| Two-site | Dense Memory Attention | `[3, 7]` | 7,344,128 |

Layers are zero-based. Groups are compared separately. The no-memory arm uses
the same writer-plus-merger parameter structure without a previous-pass input.
Memory Attention uses 16 KV heads; the Recirculation-inspired controller has
width 660. These real dimensions put every primary arm within 0.1% of its
site-count group's added-parameter budget. Generate the checked table with
`make report-wiring-budgets`; it writes the ignored, reproducible
`results/wiring_budgets.json` report.
See [architectures](../../../docs/ARCHITECTURES.md) for the mechanism contracts.

The earlier combined attention layout had readers at `[4, 7]`.
Changing them to `[3, 7]` is an architecture change, independent of the naming cleanup.
The new arm ID does not relabel old results or make that reader layout checkpoint-compatible.
The parameter-free `strided_self_attention` control is not a frozen wiring arm.

## Comparability with previous runs

This is a new experimental protocol, not a continuation or drop-in rerun of the
retired 1024-token frozen study. Packing, sequence length, optimizer batching,
update count and the combined reader layout changed. Do not overlay those curves
as a continuous trajectory or attribute cross-protocol differences to the merger alone.
All current arms share the artifact, effective batch, pass schedule,
validation, snapshot schedule and precision specified below. The combined
dense-and-strided arm uses a 4x8 microbatch/accumulation split for A6000
memory, while the other arms use 8x4; both yield the same 65,536-token
optimizer batch.

The selected BOS feedback checks are diagnostics, not a change to the K=2/3
training objective. The recorded optimizer-update timer excludes validation.
Resuming an old completed run does not backfill reports at earlier milestones;
evaluate an existing snapshot with the standalone evaluator or run a fresh trajectory.
Downstream generation remains actual-context prefill followed by ordinary feedback.

## Execution settings

These shared values mirror the runnable YAML files. Documentation tests check them.

| Config field | Value |
| --- | --- |
| `phase` | `A` |
| `data_dir` | `data/dolmino/gpu_2048` |
| `batch_size` | `8` (dense-and-strided: `4`) |
| `grad_accum_steps` | `4` (dense-and-strided: `8`) |
| `max_unique_tokens` | `100007936` |
| `dtype` | `float32` |
| `autocast_dtype` | `bfloat16` |
| `seed` | `1337` |
| `architecture_seed` | `4242` |
| `eval_passes` | `4` |
| `eval_batches` | `64` |
| `eval_every_tokens` | `3276800` |
| `train_log_every_tokens` | `327680` |
| `checkpoint_every_tokens` | `3276800` |
| `checkpoint_every_seconds` | `900` |
| `checkpoint_keep_last` | `2` |
| `feedback_eval_at_tokens` | `[5013504, 20021248, 100007936]` |
| `feedback_eval_max_blocks` | `1` |
| `feedback_eval_autocast_dtype` | `config` |

Clean blocks contain 2048 linguistic tokens. The nominal optimizer batch is 32 sequences,
or 65,536 tokens. The combined dense-and-strided arm is fixed at 4×8 for the
A6000; other arms use 8×4. If another host requires it, use 2×16 or 1×32
microbatch/accumulation after preflight. Record the resolved settings and
preserve the effective batch.

Each microbatch samples K=2 with probability 0.9 or K=3 with probability 0.1.
NTP loss uses only the final pass: `[0, 1]` or `[0, 0, 1]`.
The learning-rate schedule is constant. The reset qualification must select the
rate for each mechanism and site count before the 100M configs are launched.
The checked-in `3e-4` values are placeholders, and `STUDY.yaml` blocks training
while `learning_rates_qualified` is false. After recording the selected rates,
update the configs and set that gate to true.
The runner does not select rates or continue qualification checkpoints
automatically.

## Snapshots and evaluation

Routine K=4 validation retains K=1–4 NLL on a fixed 64-block prefix.
The trainer also evaluates the final state, reusing a matching completed check.
Full-split parallel NLL and downstream tasks require separate evaluation commands.

Snapshots occur after the optimizer update that reaches each threshold:

| Requested tokens | Actual tokens with default batching | Optimizer updates | BOS feedback NLL |
| ---: | ---: | ---: | --- |
| 3,276,800 | 3,276,800 | 50 | No |
| 5,013,504 | 5,046,272 | 77 | Yes |
| 10,027,008 | 10,027,008 | 153 | No |
| 20,021,248 | 20,054,016 | 306 | Yes |
| 50,003,968 | 50,003,968 | 763 | No |
| 100,007,936 | 100,007,936 | 1526 | Yes |

The selected checks use one common full 2048-token block with BOS-only K=1 prefill.
They run after durable snapshots, not at every routine validation.
Feedback precision currently inherits BF16. Qualify its cost on the target GPU before changing precision.
The [A6000 report](../inference_efficiency/README.md) did not time this production evaluator or both new recurrent mergers.

See [evaluation](../../../evaluation/README.md) for full/aligned target counts and contextual downstream defaults.
See [training](../../../docs/TRAINING.md) for independent recovery checkpoints,
pending-work recovery and actual snapshot counters.

Dedicated token-zero evaluation is deferred.
Compare absolute trained-checkpoint NLL, not improvement from an unmeasured initial loss.
Start with pass-depth and real/zero/mismatched/true-bypass checks on an early
trained snapshot. Interventions into passes 2, 3, and 4 are separate diagnostics,
not a pre-training gate.

## Execution and reporting

Complete the [GPU pre-training checks](../../../docs/CLOUD.md#pre-training-checks) and LR qualification first.
Then run selected primary arms sequentially. The attention extensions are
declared now but run only after the dense groups pass preflight:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_comparison --skip-wire
```

Auto-resume restores compatible existing trajectories. It starts fresh only for new outputs.
Do not reuse preflight weights or the old middle-layer recurrent outputs.

Report NLL against consumed tokens, optimizer updates, training-only seconds and estimated FLOPs.
Report end-to-end time and peak VRAM separately.
Generate the ignored, reproducible FLOP report with `make estimate-flops-frozen-backbone`.
It estimates dominant matrix operations and does not discount missing frozen-parameter gradients.
It is not a hardware FLOP counter.

Follow the [promotion rules](../../README.md) before creating a locked core study.
See the [development plan](../../../docs/DEVELOPMENT_PLAN.md) for remaining work.
