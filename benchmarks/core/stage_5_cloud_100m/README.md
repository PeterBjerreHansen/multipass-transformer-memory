# Stage 5: 100M-token cloud continuation

This locked study is the first serious eight-arm comparison after the CUDA
qualification and pilot gates.

- Every arm uses `batch_size=1` and `grad_accum_steps=1`, preserving the
  reference optimizer batch of 2,048 linguistic tokens.
- The original non-vanilla arms and Multiscale Memory Attention initialize from their final
  Stage-1 wiring checkpoints. Vanilla and Sparse SWA start from the canonical
  TinyMistral checkpoint.
- `data/dolmino/gpu_2048_staged` skips the 5,242,880-token wiring slice and
  materializes 100,007,936 new training tokens. The original `gpu_2048` recipe
  remains unchanged as the zero-offset substrate control.
- The source, data recipe, initialization checkpoints, and output paths must
  be kept unchanged once a run starts.
- Each arm keeps two durable checkpoint generations and checkpoints on both
  token and wall-clock cadences for spot recovery.

MemoryAdd and FBT are intentionally absent from this locked study. Their
historical implementations are not evidence for the current comparison.

All eight arms are complete. The two attention-control arms used the same
staged Phase-B slice and optimizer batch as the original six arms. Their full
artifacts were transferred and checksum-verified into `results/<arm>/` before
the cloud compute instance was shut down and deleted.
