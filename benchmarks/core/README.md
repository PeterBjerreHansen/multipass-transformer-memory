# Core benchmarks

This directory is reserved for larger, predeclared experiments that test the
project's central claims. It is intentionally empty until development evidence
and CUDA qualification justify a locked campaign.

Before a core run, create a study directory containing:

```text
STUDY.yaml
<runnable arm configs>.yaml
results/
```

Set `status: locked` only after the scientific question, arms, comparison axes,
data artifact, initialization provenance, and execution configs have been
reviewed. `scripts/verify_study.py` checks that compared arms differ only on the
arm-local output path, declared `experimental_axes`, and explicit
`allowed_differences`.

For the first serious dense-memory comparison, do not infer batching from GPU
capacity. The 2048-context development reference is 2,048 unique tokens per
optimizer update. Run `make efficiency-cuda-batch-qualification` on the target
GPU and use `scripts/select_cuda_batch.py` to identify the smallest common
efficient K=2 microbatch for adaptive Recirculation and dense Bank. If that recommendation
is larger than batch 1, treat the resulting optimizer-batch change as a protocol
question before locking the core study; do not automatically add gradient
accumulation to reach an arbitrary larger batch.

Core Phase-A initialization should be rerun on the same pinned data artifact as
core Phase B (`data/dolmino/gpu_2048` for the current plan), not inherited from
the `wiring_2048` development wiring checkpoints. This avoids cross-artifact
validation ambiguity while keeping the historical development lineage intact.
