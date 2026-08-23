# Data artifacts

Dataset preparation recipes live beside the generated artifacts they define.
Only the small source recipes are tracked; token binaries, manifests, and other
materialized dataset files are ignored.

- `dolmino/wiring_2048/config.yaml`: one unique 5M-token wiring epoch.
- `dolmino/pilot_2048/config.yaml`: the following unique 10M-token pilot epoch.
  It skips the full wiring slice and shares its separate 256-block validation
  set.
- `dolmino/gpu_2048/config.yaml`: separate 100M-token serious-run artifact;
  training may repeat it when a declared campaign budget exceeds one epoch.
- `dolmino/gpu_2048_staged/config.yaml`: shared validation plus non-overlapping
  wiring and Phase-B slices for initialized serious-run controls.

Use `scripts/prepare_data.py` and `scripts/verify_data.py` rather than editing a
generated artifact in place.
