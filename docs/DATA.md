# Dolmino data artifact

Training does not live-stream remote data. A one-time materialization converts a
pinned slice of `allenai/dolmino-mix-1124` into deterministic local token blocks.
Every architecture therefore starts from the same linguistic IDs and block
order.

The checked-in preparation recipes live beside their local generated
artifacts:

```text
data/dolmino/wiring_2048/config.yaml
data/dolmino/gpu_2048/config.yaml
data/dolmino/unfrozen_lr_20m_2048/config.yaml
data/dolmino/unfrozen_2p5b_2048/config.yaml
data/dolmino/unfrozen_final_eval_2048/config.yaml
```

`gpu_2048` is the active frozen-study artifact: 2048-token blocks, approximately
100M training tokens and 2M validation tokens. `wiring_2048` is the small
pre-training and wiring-check artifact. The unfrozen LR artifact contains about
20M training tokens. The scaling artifact contains exactly 2,500,001,792
training tokens; feedback arms traverse it once and vanilla cycles it to its
compute-matched endpoint. The independent final-evaluation validation split is
10,027,008 tokens after a 3,000,000,512-token stream offset from the same pinned
source recipe.

Evaluation now records the selected prefix blocks, split and manifest identity,
block lengths and target counts. Do not assume
that the training monitoring split is the independent evaluation artifact, or
that adding BOS leaves scored-token coverage unchanged. See
[the evaluation contract](../evaluation/README.md).

BOS-only feedback prepends a context BOS without altering packed data. It scores all data tokens and separately reports the aligned
score excluding the first token; it does not replace a block's first token with BOS.

Generated binaries and manifests remain local/ignored.

## Split construction

For each source, validation documents are consumed first from a deterministic
shuffled streaming iterator; the final partially used validation document is
discarded. Training then continues from the next document. Thus train and
validation are source-document-disjoint **within one materialized artifact**.
Optional `validation_skip_tokens` and `train_skip_tokens` consume and discard
complete source-balanced blocks before the corresponding split. The final
unfrozen evaluation recipe uses the former to move beyond the long-run training
range. The active training recipes use no skip. Verify complete-document
disjointness after all unfrozen artifacts are materialized; stream offsets do
not replace that check.

Documents are tokenized with the pinned TinyMistral tokenizer after explicitly
disabling its persisted padding and truncation settings. The Dolmino recipe
rewrites a literal `[PAD]` source string as `[ PAD ]` before tokenization so
source text cannot become the reserved padding ID. The manifest records this
text-normalization policy and replacement. BOS is used as an explicit document
separator. The packer owns the fixed 2048-token boundary, so published blocks
contain raw document IDs plus BOS separators, not tokenizer padding. New
artifacts use manifest format 2 and the `raw_unpadded_document_stream_v1`
packing policy.

## On-disk format

```text
artifact/
  train.bin
  train.sources.bin
  validation.bin
  validation.sources.bin
  manifest.json
```

The binary token IDs are ordinary vocabulary IDs only. The manifest records the
vocabulary size, source allocation, tokenizer hash, requested/resolved dataset
revision, recipe, shuffle settings, seed, split-stream offsets, forbidden
control-token IDs, text-normalization policy, replacements, and file hashes.
Verification scans the token files while checking their hashes and rejects any
recorded padding/control ID. Training and
the packed-data evaluation CLIs run this complete verification before consuming
an artifact; lightweight dataset construction still checks the manifest
contract and expected file sizes.

## Active split ownership

The document-disjoint guarantee belongs to one materialized artifact.
`wiring_2048` is reserved for wiring and pre-training checks. The larger
`gpu_2048` artifact is the sole source for the active frozen studies and GPU
substrate control. The unfrozen studies use their purpose-specific artifacts.
A run may reshuffle and repeat its assigned slice when its budget exceeds the
stored token count; this is the intended vanilla compute-matched tail.

The within-artifact property does not prove disjointness across separately
materialized artifacts. Check the unfrozen final-evaluation artifact against
the LR and long-run training artifacts with
`scripts/verify_data_disjointness.py` before retaining final results.

The disjointness tool compares exact complete BOS-delimited tokenized documents.
It excludes leading and trailing fragments. It does not detect every shared
passage or near-duplicate document. A passing report is not proof of no possible
contamination. Retain the artifact identities and the report's coverage limits.
