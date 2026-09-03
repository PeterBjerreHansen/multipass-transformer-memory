# Training contract

This document defines reusable trainer semantics. Scientific protocol decisions
belong under `benchmarks/development/` and, once locked, under
`benchmarks/core/`.

## Research stages and trainer phases

The trainer has two mechanics:

- **Phase A:** pretrained TinyMistral parameters are frozen; only
  architecture-added parameters train.
- **Phase B:** the full model is differentiable, with separate pretrained and
  added-parameter optimizer groups.

Phase names describe optimization mechanics, not research-stage names.

An integrated retrofit remains one Phase-B trajectory. Set
`freeze_pretrained_until_tokens` to train only architecture-added parameters at
the beginning of that trajectory. At the boundary, the trainer enables the
pretrained parameters without rebuilding the optimizer. The added group keeps
its AdamW moments and step count; the pretrained group has no optimizer state
until it first receives a gradient. The threshold must align to a complete
microbatch, and an optimizer update is shortened when necessary so it cannot
straddle the boundary.

Use Phase A, not an integrated freeze threshold, when the backbone should stay
frozen for the entire run.

## Pass objective

For pass losses `L_1 ... L_K`, configured non-negative loss weights are
right-aligned to K and normalized:

```text
L = sum_k w_k L_k
```

`pass_schedule` independently controls the sampled pass count. Its RNG/state is
checkpointed, so fixed or mixed K schedules resume exactly.

The pass scheduler samples K per microbatch, not per optimizer update or token.
See the [study protocols](../benchmarks/README.md) for probabilities and loss weights.
Historical Phase-B objectives do not define the future unfrozen study.

## Supported training execution

`training_forward: parallel_multipass` computes complete sequence passes;
`pass_schedule` and NTP loss weights define the objective. Paper readout/replay
and its BPTT/TBPTT implementation have been removed, along with their truncation
and activation-checkpointing settings. Ordinary preceding-token feedback and
the adaptive recirculation merger remain supported.

## Parameter groups and schedules

Phase B maintains separate AdamW groups for pretrained and architecture-added
parameters. Each group has its own base learning rate. A common schedule
multiplier is supplied by `constant`, `cosine`, or `piecewise_linear` schedule
logic.

`pretrained_weight_decay` and `added_weight_decay` optionally separate the two
groups' AdamW decay. If either is omitted, it inherits `weight_decay`. This
allows a common-checkpoint retrofit to retain the backbone's established decay
while reproducing a controller-specific paper setting.

## `init_from`, `resume_from`, and auto-resume

`init_from` loads model weights only and starts a new optimizer/data/RNG
trajectory. `resume_from` restores the same trajectory: model, optimizer,
sampler, pass scheduler, RNG, counters, phase, and compatible config.

For spot/cloud operation, `scripts/train.py --resume-auto` starts fresh only
when the run directory is empty; otherwise it resumes the newest valid durable
checkpoint generation. Ambiguous existing state fails closed. See `CLOUD.md`.

## Batching and token budgets

Microbatch size and optimizer-batch size remain distinct. For ordinary packed
data with L linguistic tokens per block:

```text
linguistic tokens / microbatch = batch_size * L
linguistic tokens / nominal optimizer update = batch_size * grad_accum_steps * L
```

A larger hardware microbatch can therefore change the scientific optimizer
batch when accumulation is unchanged. It must not be treated as a harmless
hardware default.

The active frozen studies use 2048-token blocks, microbatch 8 and accumulation
4, for 32 sequences per nominal optimizer update. Optional hardware tuning can
use 4x8, 2x16 or 1x32 when needed. Preserve the effective batch and record the
resolved config. Each mechanism selects its own LR from the common qualification
budget. No replacement unfrozen protocol is active yet.

The trainer consumes whole packed blocks. An exact linguistic-token budget must
be divisible by `batch_size * linguistic_tokens_per_block`; the final optimizer
update may use fewer accumulation microsteps, but blocks are never cropped.

## Memory-token accounting

The backing dataset contains only linguistic tokens. In
`memory_write_mode: memory_token`, a deterministic view inserts physical MEM
positions. A backing block of N linguistic tokens and cadence C becomes

```text
P = N + floor((N - 1) / C)
```

physical positions. Thus the trainer records separate counters:

```text
unique_tokens_seen       += linguistic tokens
model_positions_seen     += physical positions
token_equivalent_compute += physical positions * effective passes
```

`max_unique_tokens`, pass scheduling, and LR scheduling use linguistic tokens.
Training metrics additionally report linguistic tokens/s and model positions/s.
This keeps data dose comparable while making the extra MEM computation visible.
Despite its name, `unique_tokens_seen` counts consumed linguistic positions.
It is not a corpus-wide deduplication counter. The sampler reshuffles and repeats
the artifact if the requested budget exceeds its stored training split.

When `train_log_every_tokens` aggregates multiple optimizer updates, throughput
is total interval tokens divided by total measured optimizer-update time. Pass
losses use per-key observation counts, so a K=3-only metric is not diluted by
K=2 updates. Each record includes those counts, an interval pass histogram, and
the interval duration. A graceful signal flushes the unfinished interval with
`log_interval_partial: true` before checkpointing.

See [Memory Attention](MEMORY_ATTENTION.md) for MEM labels and the Phase-A
embedding gradient path. See [evaluation precision](../evaluation/README.md#precision)
for FP32 storage, autocast and overrides.

## Checkpoint cadence and validation

Serious cloud runs may checkpoint on either a linguistic-token cadence or a
wall-clock cadence. New trained state is checkpointed before expensive periodic
validation, so an interruption during validation does not lose the preceding
optimizer work. Validation is read-only and should not implicitly alter the
training protocol.

`checkpoint_keep_last: 1` is supported for local runs where disk use matters;
it retains only the newest verified generation and therefore cannot fall back
if that file later becomes unreadable. Keep the default value `2` for cloud or
other interruption-sensitive runs.

Optional `snapshot_at_tokens` writes weights-only safetensors for scientific
analysis. A threshold is served at the first completed optimizer update that
reaches it. Thresholds crossed in the same update share one snapshot, with all
requested thresholds and the actual token count recorded.

The trainer first commits a resumable checkpoint with pending snapshot work.
It then writes and fsyncs a temporary safetensors file containing both weights
and metadata, atomically replaces the destination, and fsyncs the directory.
The JSON sidecar is a repairable mirror, not a completion marker. On restart,
pending work finishes at the recovered model state before training advances.
Retries verify existing committed weights and identity rather than overwrite
a different model at the same path.

New snapshots contain their experiment config and can be loaded without their
original run directory or sidecar. Legacy snapshots still require their
sidecar and parent `run.json`. Planned snapshots are never pruned by rolling
checkpoint retention. Snapshots never drive resume: generation checkpoints
remain the optimizer/data/RNG trajectory source of truth.

Training logs record input tokens, model positions, token-equivalent compute,
interval time, cumulative `training_elapsed_seconds`, and the realized pass
schedule. The cumulative counter measures synchronized optimizer-update work
and is checkpointed across resumes; validation and checkpoint I/O are excluded.
Use one trajectory per arm, then rescale the x-axis for data, estimated compute,
or training-time views during analysis. Do not rerun an arm solely to produce a
different plot axis.

## Evaluation

Trainer validation and standalone NLL now call the same parallel evaluator,
with explicit experiment autocast, model-owned labels, token-weighted per-pass
and per-source losses, and subset/precision metadata. Routine block limits and
training cadence are unchanged. See [the evaluation contract](../evaluation/README.md).

The active frozen study uses 64 blocks per routine check. Standalone evaluation
defaults to the full split, so specify the same block limit when comparing it
with trainer results. Expensive feedback diagnostics do not define the routine
validation metric or its early-stopping decisions.

## Selected-checkpoint feedback validation

Feedback NLL is disabled unless `feedback_eval_at_tokens` is nonempty.
These thresholds must be a subset of `snapshot_at_tokens`.
`feedback_eval_max_blocks` sets the number of complete prefix blocks.
`feedback_eval_autocast_dtype` selects the [evaluation precision](../evaluation/README.md#precision).
The [main frozen protocol](../benchmarks/development/frozen_backbone_comparison/README.md)
documents the enabled milestones. The LR sweep leaves this schedule off.

After crossing a selected threshold, the trainer commits resumable state,
publishes the durable snapshot, then evaluates full-block BOS-only feedback NLL
before the next optimizer update. This is a synchronous validation pause, not a
background worker. It is outside measured optimizer time and does not change
routine K=4 cadence, LR selection metrics or early stopping. Several selected
thresholds crossed by one update share one report.

Reports live in `evaluations/feedback_model_<actual-tokens>_<request-hash>.json`.
They identify snapshot bytes, evaluator source/runtime, precision, data prefix, BOS
initialization and target coverage. `feedback_validation` journal events are
separate from routine `validation` events. Full and aligned NLL report 2048 and
2047 targets per ordinary block respectively; see [evaluation](../evaluation/README.md).

Pending snapshot work includes its selected feedback check. A graceful signal
during decoding returns without a partial metric; an interrupted check reruns
from the block start on resume, before training advances. A complete matching
report is reused, and a missing journal event is repaired without decoding
again. Changed precision, subset or evaluator source/runtime produces a separate report.
Rolling checkpoint retention never removes snapshots or feedback reports.

Schedule edits apply to future thresholds and work still pending in the resumed
checkpoint. They do not scan older snapshots retroactively. Use the standalone
command for an older snapshot. Editing a schedule does not launch a run.
