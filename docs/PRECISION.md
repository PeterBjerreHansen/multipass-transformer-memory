# Precision contract

Trainer validation and standalone NLL, pass-depth, continuation, intervention,
and downstream evaluation now use the same evaluation context. The command-line
default inherits the experiment's `autocast_dtype`; `--autocast-dtype float32`
disables autocast and `--autocast-dtype bfloat16` requests BF16 explicitly.
Overrides do not change the training config used to identify a checkpoint.

Selected-checkpoint BOS feedback uses `feedback_eval_autocast_dtype`:
`config` (default), `float32` or `bfloat16`. This can differ explicitly from
routine validation precision. Reports record both the actual setting and target
coverage; a precision change creates a separate durable report on retry.

Results record the resolved device, parameter dtypes, autocast setting and FP32
loss reduction. Compare scores at the same precision. A BF16 CUDA experiment
evaluated on CPU needs an explicit `--autocast-dtype float32` override; unsupported
precision requests fail rather than silently changing compute mode.

Direct Python evaluator calls have no experiment config: pass `autocast_dtype`
explicitly, or use their default of no autocast. The context restores model
training mode on success and failure. Hardware speed and numerical qualification
remain separate from these shared execution rules.

`dtype` is the **learned-parameter storage dtype**. Existing MPS development
runs remain FP32.

For serious CUDA training the intended mode is:

```yaml
device: cuda
dtype: float32
autocast_dtype: bfloat16
```

This keeps learned parameters and ordinary AdamW state in FP32 while executing
eligible CUDA operations under BF16 autocast. BF16 here is a training compute
format, not INT8/INT4 inference quantization.

MPS uses FP32 parameter storage and FP32 compute by default. The efficiency
battery also tests:

```yaml
device: mps
dtype: float32
autocast_dtype: bfloat16
```

MPS BF16 autocast is treated as a **capability-dependent engineering mode**:
PyTorch/macOS support can vary by host stack. The precision benchmark records it
as `unsupported` if the requested mode is unavailable. It should not be promoted
to a scientific training protocol until the local precision comparison is finite
and stable.

The experiment configuration deliberately allows only BF16 autocast for now.
FP16 training would require a separately validated loss-scaling policy and is
not silently enabled.

Pure BF16 parameter storage is not the intended training contract. The key
comparison is FP32 parameters/optimizer state with either FP32 compute or BF16
autocast compute.
