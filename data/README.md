# Data artifacts

Dataset preparation recipes live beside the generated artifacts they define.
Only the small source recipes are tracked; token binaries, manifests, and other
materialized dataset files are ignored.

The 1024-token study and the former staged-pipeline recipes have been deleted.
The active data namespace contains:

- `dolmino/wiring_2048/config.yaml`: the small, one-epoch wiring and
  pre-training check (5M training tokens and 0.5M validation tokens).
- `dolmino/gpu_2048/config.yaml`: the canonical 100M-token artifact for the
  frozen comparison, LR qualification, and GPU substrate control.
- `dolmino/unfrozen_lr_20m_2048/config.yaml`: the 20M-token Phase-B LR slice.
- `dolmino/unfrozen_2p5b_2048/config.yaml`: the common 2.5B-token long-run
  corpus; the vanilla arm repeats this artifact rather than receiving new data.
- `dolmino/unfrozen_final_eval_2048/config.yaml`: a 10M-token held-out
  validation slice beginning after a 3B-token stream offset.

All previously materialized binaries were removed after the tokenizer-padding
audit. The old pilot, staged-100M, historical 2.5B continuation, and Stage-6
evaluation recipes are no longer part of the current contract. The new
unfrozen recipes are fresh definitions and do not reactivate those protocols.
Use `scripts/prepare_data.py` and `scripts/verify_data.py` rather than editing a
generated artifact in place.

Preparation explicitly disables the tokenizer's persisted padding and
truncation settings before encoding. The packer, not the tokenizer, owns the
2048-token boundary. New manifests use the raw unpadded packing policy; do not
reuse an artifact with an older manifest format. Verification scans the token
files and requires zero occurrences of the tokenizer's recorded padding ID.
The Dolmino recipe also rewrites a literal `[PAD]` source string as
`[ PAD ]` before tokenization. This is recorded in the manifest's text
normalization fields; it does not enable tokenizer padding.
