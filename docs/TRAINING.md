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

The original multipass study samples K=2 on 90% of batches and K=3 on 10%.
K-specific loss weights make Phase A estimate `0.9 L2 + 0.1 L3` with 2.1
average passes. Phase B also retains a first-pass loss. One-pass controls such
as Strided Attention use K=1. Runnable protocols live under `benchmarks/development/`.

## Recirculation training forwards

`training_forward: parallel_multipass` is the existing whole-block objective.
It computes K complete sequence passes, and `pass_schedule` plus the NTP loss
weights define the training objective.

`training_forward: recirculation_bptt` is a different computation, not K=2.
For each token it:

1. performs the ordinary first iteration and reads out its logits;
2. mixes the source and destination residuals;
3. replays the token above the destination layer and replaces those KV entries;
4. lets the next token attend to the replayed strict-past cache.

The `pass_schedule` must therefore be K=1 and multipass loss weights are
invalid. `recirculation_activation_checkpointing: true` recomputes token steps
during backward to reduce saved activations without changing gradients.

By default, `recirculation_bptt_truncate_tokens: null` propagates gradients
through the complete packed sequence. This is the paper-faithful BPTT policy.
A positive value selects explicit TBPTT: the forward KV values continue across
the complete sequence, but the cache is detached every N input positions. The
trainer backpropagates each chunk before computing the next one, so completed
graphs can be released. A finite window changes the gradient estimator and must
be qualified and reported; it is not a harmless hardware switch.

The legacy `effective_passes` and `token_equivalent_compute` counters record two
recurrence iterations for this forward. The second iteration replays only the
layers above the destination, not the full backbone. Training records therefore
also include the replayed-layer count and fraction. Use those fields with the
FLOP estimator—or measured accelerator time—instead of interpreting the legacy
2x counter as a FLOP ratio.

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

The planned recirculation studies target 32 sequences per optimizer update. A
protocol may use microbatch 1 with accumulation 32, microbatch 2 with
accumulation 16, and so on, but it must state whether physical batching is
fixed or hardware-tuned. The frozen-backbone comparison fixes microbatch 1 and
accumulation 32 for every arm. Preserve `batch_size * grad_accum_steps` when a
different study changes only hardware fit. Learning rate still requires
qualification for this model and optimizer batch; copying a paper value is a
starting hypothesis, not evidence that it is optimal.

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

When `train_log_every_tokens` aggregates multiple optimizer updates, throughput
is total interval tokens divided by total measured optimizer-update time. Pass
losses use per-key observation counts, so a K=3-only metric is not diluted by
K=2 updates. Each record includes those counts, an interval pass histogram, and
the interval duration. A graceful signal flushes the unfinished interval with
`log_interval_partial: true` before checkpointing.

## Memory-token loss

MEM uses input ID V, where V is the base vocabulary size, but the LM head remains
V classes. The Memory Attention model constructs position-aligned labels over linguistic
tokens. For:

```text
physical: A  <MEM>  B  C
labels:   B  IGNORE C  IGNORE
```

A predicts B; MEM predicts nothing. See `MEMORY_ATTENTION.md` for the public
vocabulary and `BANK_MEMORY.md` for the full compatibility contract.

## Phase A and MEM

Dense and strided Memory Attention Phase A can discard the frozen pass-1 graph because no
added parameter occurs in pass 1. Memory-token Phase A cannot: the added MEM
embedding participates in pass 1 and must receive gradient through later
recurrent/Memory Attention loss. The backbone remains frozen, but pass-1 autograd stays
enabled for that mode.

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
analysis. Snapshots never drive resume; resumable generation checkpoints remain
the trajectory source of truth.

Training logs record input tokens, model positions, token-equivalent compute,
interval time, cumulative `training_elapsed_seconds`, and the realized pass
schedule. The cumulative counter measures synchronized optimizer-update work
and is checkpointed across resumes; validation and checkpoint I/O are excluded.
Use one trajectory per arm, then rescale the x-axis for data, estimated compute,
or training-time views during analysis. Do not rerun an arm solely to produce a
different plot axis.
