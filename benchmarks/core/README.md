# Core benchmarks

This directory is reserved for larger, predeclared experiments that test the
project's central claims. Keep development protocols under `development/`
until their evidence and CUDA qualification justify promotion.

Before a core run, create a study directory containing:

```text
STUDY.yaml
<runnable arm configs>.yaml
results/
```

The current locked Stage-5 eight-arm study is in
`benchmarks/core/stage_5_cloud_100m/`. All eight arms are complete; their
full, transferred artifacts are stored under the ignored `results/<arm>/`
directories.

Set `status: locked` only after the scientific question, arms, comparison axes,
data artifact, initialization provenance, and execution configs have been
reviewed. `scripts/verify_study.py` checks that compared arms differ only on the
arm-local output path, declared `experimental_axes`, and explicit
`allowed_differences`.

For a serious comparison, do not infer batching from GPU capacity. The
2048-context reference is 2,048 unique tokens per
optimizer update. Run `make efficiency-cuda-batch-qualification` on the target
GPU and use `scripts/select_cuda_batch.py` to identify the smallest common
efficient K=2 microbatch for adaptive Recirculation and dense Bank. If that recommendation
is larger than batch 1, treat the resulting optimizer-batch change as a protocol
question before locking the core study; do not automatically add gradient
accumulation to reach an arbitrary larger batch.

For an initialized control, use declared non-overlapping Phase-A and Phase-B
slices with one held-out validation set, as provided by
`data/dolmino/gpu_2048_staged`.
