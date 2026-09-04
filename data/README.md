# Data artifacts

Dataset preparation recipes live beside the generated artifacts they define.
Only the small source recipes are tracked; token binaries, manifests, and other
materialized dataset files are ignored.

The 1024-token study and the former staged-pipeline recipes have been deleted.
Only two recipes remain in the active data namespace:

- `dolmino/wiring_2048/config.yaml`: the small, one-epoch wiring and
  pre-training check (5M training tokens and 0.5M validation tokens).
- `dolmino/gpu_2048/config.yaml`: the canonical 100M-token artifact for the
  frozen comparison, LR qualification, and GPU substrate control.

All previously materialized binaries were removed after the tokenizer-padding
audit. The old pilot, staged-100M, 2.5B continuation, and Stage-6 evaluation
recipes are no longer part of the repository contract. Their benchmark files
remain only as historical provenance and are not runnable current studies.
Use `scripts/prepare_data.py` and `scripts/verify_data.py` rather than editing a
generated artifact in place.

Preparation explicitly disables the tokenizer's persisted padding and
truncation settings before encoding. The packer, not the tokenizer, owns the
2048-token boundary. New manifests use the raw unpadded packing policy; do not
reuse an artifact with an older manifest format. Verification scans the token
files and requires zero occurrences of the tokenizer's recorded padding ID.
