# Data artifacts

Dataset preparation recipes live beside the generated artifacts they define.
Only the small source recipes are tracked; token binaries, manifests, and other
materialized dataset files are ignored.

The 1024-token study recipe has been deleted. `gpu_2048` serves the active
frozen comparison and LR sweep; the other recipes retain their separate roles:

- `dolmino/wiring_2048/config.yaml`: one unique 5M-token wiring epoch.
- `dolmino/pilot_2048/config.yaml`: the following unique 10M-token pilot epoch.
  It skips the full wiring slice and shares its separate 256-block validation
  set.
- `dolmino/gpu_2048/config.yaml`: separate 100M-token serious-run artifact;
  training may repeat it when a declared campaign budget exceeds one epoch.
- `dolmino/gpu_2048_staged/config.yaml`: shared validation plus non-overlapping
  wiring and Phase-B slices for initialized serious-run controls.
- `dolmino/gpu_2048_long_2p5b/config.yaml`: the pinned 2.5B-token continuation
  recipe for the Stage-6 long plastic study; it preserves the 5M wiring offset
  and the shared validation construction while extending the training stream
  beyond the Stage-5 100M artifact.
- `dolmino/stage_6_evaluation_2048/config.yaml`: an independent Stage-6
  evaluation stream placed after the complete long-run source range. Its
  materialized validation split must still pass the document-disjointness gate.
Use `scripts/prepare_data.py` and `scripts/verify_data.py` rather than editing a
generated artifact in place.
