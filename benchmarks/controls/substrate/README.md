# Substrate controls

Vanilla TinyMistral reference runs for the local Mac and GPU environments.

- `mac.yaml`: local 2048-token control.
- `gpu.yaml`: long GPU preflight/control configuration. It starts with
  `batch_size=1` and `grad_accum_steps=1`, which preserves the validated
  2,048-token optimizer batch.

Raw outputs are written under `results/<arm>/` and ignored. Retain only compact
comparison notes when a substrate result matters scientifically.
