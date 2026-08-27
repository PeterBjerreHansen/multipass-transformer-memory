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
as Sparse SWA use K=1. Runnable protocols live under `benchmarks/development/`.

## Parameter groups and schedules

Phase B maintains separate AdamW groups for pretrained and architecture-added
parameters. Each group has its own base learning rate. A common schedule
multiplier is supplied by `constant`, `cosine`, or `piecewise_linear` schedule
logic.

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
