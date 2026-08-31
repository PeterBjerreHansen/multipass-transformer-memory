# Efficiency benchmarks

These benchmarks characterize implementation efficiency: wall time, throughput,
memory use, precision mode, feasible microbatch size, and context scaling. They
are **engineering measurements**, not scientific model-quality evidence, and do
not belong under `benchmarks/development/` or `benchmarks/core/`.

The shared runner performs real forward/backward/AdamW optimizer steps on
deterministic synthetic token IDs so data loading and storage do not contaminate
the measurement. It supports explicit gradient accumulation and reports both
hardware-facing microbatch quantities and optimizer-facing batch quantities.

For sequence length `T`:

```text
microbatch_tokens = batch_size * T
optimizer_batch_tokens = batch_size * grad_accum_steps * T
```

Changing `batch_size` can therefore be both an engineering change and a
scientific optimizer-batch change. The benchmark tooling reports the distinction
rather than hiding it.

## Recirculation forward and truncation qualification

Run the paper-forward engineering suite before selecting a BPTT configuration:

```bash
uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/forward_modes.yaml \
  --output benchmarks/efficiency/results/cuda_forward_modes.json
```

Every recurrent case keeps the 1,024-token forward sequence fixed. The suite
compares full BPTT with TBPTT windows 128, 256, and 512, all using activation
checkpointing, plus whole-block Recirculation and dense Memory Attention
references. A TBPTT row carries its window explicitly in the JSON. Its loss is
forward-equivalent, but its gradients are truncated at window boundaries.

The suite starts with hardware microbatch 1. If the GPU can fit more, repeat the
feasible window at larger microbatches. For a scientific optimizer batch of 32
sequences, reduce gradient accumulation as microbatch increases. Peak memory
and runtime choose feasible execution settings; held-out NLL and short training
curves choose among gradient windows and learning rates.

Afterward, run `make estimate-flops-forward-modes`. The estimator counts the
ordinary stack, only the layers above the recirculation destination on replay,
one LM-head readout, the adaptive controller, and activation-checkpoint
recomputation. It conservatively does not discount the small backward savings
at TBPTT boundaries.

For token-diagonal recurrence, `stack_iterations_per_token: 2` means one
ordinary readout plus one optimized upper-stack replay. It is not two complete
backbone passes. Use measured runtime or an architecture-aware FLOP estimator,
not this iteration count, for compute-axis claims.

## Relative training FLOP estimates

The README results table uses a configuration-aware relative training-FLOP
estimate rather than the measured A6000 runtime multiplier. Generate the Stage-5
estimate with:

```bash
make estimate-flops-stage5
```

The estimator reads the model configuration and counts the dominant matrix
products for each full-sequence optimizer step: backbone Q/K/V/O projections,
local self-attention score/value products, SwiGLU projections, every pass's
LM-head projection, Memory Attention writer/reader projections and attention, and adaptive
recirculation controller matrices. It accounts for the exact physical sequence
length and memory-write positions, including the 2,111-position Memory-token
sequence produced from 2,048 linguistic tokens. K=2 and K=3 are combined with
the study's 90%/10% schedule.

The convention is two FLOPs per multiply-add and 3x forward FLOPs for training
to approximate the forward plus input/weight-gradient matmuls. LayerNorm,
activations, softmax, RoPE, masking/gathering, residual operations, embedding
lookups, and optimizer arithmetic are excluded rather than assigned arbitrary
costs. The output is therefore a reproducible dominant-matmul estimate, not a
hardware-instruction trace or a wall-clock prediction. Runtime measurements
remain useful for GPU-specific scheduling decisions and are retained in the
JSON efficiency results.

## Serious CUDA batching qualification

The 2048-token development evidence was trained with `batch_size=1` and
`grad_accum_steps=1`, i.e. 2,048 unique tokens per optimizer update. The current
CUDA substrate config intentionally starts from that same optimizer-batch size.
It does **not** assume that 32k tokens/update is appropriate merely because a GPU
can process larger batches.

On the intended GPU, run:

```bash
make efficiency-cuda-batch-qualification
make select-cuda-batch \
  RESULT=benchmarks/efficiency/results/cuda_batch_qualification.json
```

The qualification suite tests K=2 adaptive Recirculation and dense Memory Attention at 2048 context,
FP32 parameter/optimizer storage, BF16 autocast, `grad_accum_steps=1`, and
microbatches 1/2/4/8. OOM cases are recorded rather than aborting the suite.

`select_cuda_batch.py` chooses the **smallest common successful microbatch** that
reaches at least 90% of each architecture's best throughput across all of that
architecture's feasible tested batches. The selected batch must still be common
to every requested architecture. This biases toward preserving the smaller
optimizer batch when extra batching buys little throughput. Its output explicitly
states whether the recommendation changes the 2,048-token reference optimizer
batch. If it does, that is a protocol question to qualify before locking a core
run.

Gradient accumulation should not be increased just to manufacture a larger
optimizer batch. Use it only when a scientifically chosen optimizer batch must
be implemented with a smaller hardware microbatch.

## General suites

The shared training, batch-scaling, and precision suites use 2048-token context.
The context-scaling suite also retains 512, 4096, and 8192 as explicit
engineering comparison points; those are not default training settings.

### MPS

```bash
uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/training.yaml \
  --device mps \
  --output benchmarks/efficiency/results/mps_training.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/precision_mps.yaml \
  --output benchmarks/efficiency/results/mps_precision.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/context_scaling.yaml \
  --device mps \
  --output benchmarks/efficiency/results/mps_context.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/batch_scaling.yaml \
  --device mps \
  --output benchmarks/efficiency/results/mps_batch.json
```

MPS context and batch scaling use FP32 compute by default. If the precision
suite shows that BF16 autocast is supported and numerically healthy on the
specific Mac/PyTorch stack, the general suites can also be rerun with
`--autocast-dtype bfloat16`.

### CUDA

```bash
uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/training.yaml \
  --device cuda \
  --output benchmarks/efficiency/results/cuda_training.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/precision_cuda.yaml \
  --output benchmarks/efficiency/results/cuda_precision.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/context_scaling.yaml \
  --device cuda --autocast-dtype bfloat16 \
  --output benchmarks/efficiency/results/cuda_context.json

uv run python scripts/benchmark_training_efficiency.py \
  --suite benchmarks/efficiency/suites/batch_scaling.yaml \
  --device cuda --autocast-dtype bfloat16 \
  --output benchmarks/efficiency/results/cuda_batch.json
```

Each successful row records linguistic tokens/s, physical model positions/s, pass-positions/s, optimizer steps/s,
microbatches/s, milliseconds/optimizer-step, microbatch tokens,
optimizer-batch tokens, projected hours per 100M unique tokens,
parameter/gradient/optimizer-state dtypes, and available memory telemetry. CUDA
reports peak allocated/reserved memory. MPS currently reports allocator/driver
memory at the end of the measured window because PyTorch does not expose the
same peak API.

### Stage-5 architecture comparison

For a measured comparison of the exact eight 100M study architectures, run:

```bash
make efficiency-cuda-stage5
```

This suite measures K=1 for the SWA Transformer and K=2/K=3 for each multipass method at
2048 tokens, batch size 1, FP32 parameter/optimizer storage, and BF16 autocast.
It includes the study's actual Memory Attention reader layers and adaptive recirculation
source/destination layers. Combine the K=2 and K=3 rows using the study's
90%/10% pass schedule to estimate wall-clock hours per 100M linguistic tokens;
do not use the nominal pass-count multiplier as a substitute for this measured
quantity.

A case that exceeds memory is recorded as `status: oom` and the suite continues.
An unavailable BF16 mode is recorded as `status: unsupported` rather than
invalidating the whole precision suite.

Compact retained results live under `benchmarks/efficiency/results/`. See
`docs/PRECISION.md` for the training precision contract.

## Memory Attention write-scaling suite

`benchmarks/efficiency/suites/bank_write_scaling.yaml` compares dense Memory Attention and
strided C=1/4/8/16/32 at W=32.
These are engineering measurements only; they do not select the scientific
write cadence.

## Attention-control suite

`benchmarks/efficiency/suites/attention_controls.yaml` remains a focused subset
of the locked Stage-5 suite: one-pass `strided_attention`, multipass
`multiscale_memory_attention`, the strided Memory Attention endpoint, and the existing
Recirculation–Strided Memory Attention hybrid. The complete eight-arm suite is
`stage_5_architectures.yaml`. Run either with the shared efficiency runner or
pass it to `scripts/estimate_training_flops.py` for dominant-matmul accounting.

For ordinary/dense/strided cases, `sequence_length` is both the linguistic and
physical length. If a future efficiency case uses `memory_token`, the runner
interprets `sequence_length` as linguistic length, inserts deterministic MEM
positions into the synthetic block, and reports the resulting
`model_sequence_length` plus separate physical-position throughput.
