# Dolmino data artifact

Training does not live-stream remote data. A one-time materialization converts a
pinned slice of `allenai/dolmino-mix-1124` into deterministic local token blocks.
Every architecture therefore starts from the same linguistic IDs and block
order.

The two checked-in preparation recipes live beside their local generated
artifacts:

```text
data/dolmino/wiring_2048/config.yaml
data/dolmino/gpu_2048/config.yaml
```

`gpu_2048` is the active frozen-study artifact: 2048-token blocks, approximately
100M training tokens and 2M validation tokens. `wiring_2048` is the small
pre-training and wiring-check artifact. The former pilot, staged, long-run, and
Stage-6 evaluation recipes were deleted during the clean-slate data reset; the
corresponding benchmark files are archival provenance only, not active runnable
studies.

Evaluation now records the selected prefix blocks, split and manifest identity,
physical/linguistic lengths, control cadence and target counts. Do not assume
that the training monitoring split is the independent evaluation artifact, or
that adding BOS leaves scored-token coverage unchanged. See
[the evaluation contract](../evaluation/README.md).

BOS-only feedback prepends a context BOS without altering packed data or MEM
insertion cadence. It scores all data tokens and separately reports the aligned
score excluding the first token; it does not replace a block's first token with BOS.

Generated binaries and manifests remain local/ignored.

## Split construction

For each source, validation documents are consumed first from a deterministic
shuffled streaming iterator; the final partially used validation document is
discarded. Training then continues from the next document. Thus train and
validation are source-document-disjoint **within one materialized artifact**.
An optional `train_skip_tokens` consumes and discards complete source-balanced
blocks after validation and before the stored training split. The active
recipes use no skip; a future purpose-specific artifact may use this field only
when its split ownership is documented and verified.

Documents are tokenized with the pinned TinyMistral tokenizer after explicitly
disabling its persisted padding and truncation settings. BOS is used as an
explicit document separator. The packer owns the fixed 2048-token boundary, so
published blocks contain raw document IDs plus BOS separators, not tokenizer
padding. New artifacts use manifest format 2 and the
`raw_unpadded_document_stream_v1` packing policy.

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
control-token IDs, and file hashes. Verification scans the token files while
checking their hashes and rejects any recorded padding/control ID. Training and
the packed-data evaluation CLIs run this complete verification before consuming
an artifact; lightweight dataset construction still checks the manifest
contract and expected file sizes.

## Memory-token data view

Explicit `<MEM>` positions are **not** written into the stored Dolmino artifact
and do not require tokenizer mutation. `MemoryTokenPackedDataset` wraps the
ordinary artifact at load time and inserts control ID V, where V is the base
vocabulary size.

For N linguistic tokens and cadence C:

```text
physical positions = N + floor((N - 1) / C)
```

No trailing MEM is inserted after the final linguistic token because there is no
following linguistic token inside that block. Ordinary token order and source ID
are unchanged.

This means every standard backing block remains 2048
**linguistic** tokens. At C=8 the Memory Attention model processes 2303 physical positions.
That extra compute is intentional and separately accounted; it avoids silently
reducing the text/data dose for MEM experiments.

The model's maximum position range must fit the expanded block.
The trainer validates this before training. The cloud preflight reports expanded
batching but does not replace the trainer's checks.

## Active split ownership

The document-disjoint guarantee belongs to one materialized artifact.
`wiring_2048` is reserved for wiring and pre-training checks. The larger
`gpu_2048` artifact is the sole source for the active frozen studies and GPU
substrate control. A run may reshuffle and repeat its assigned slice when its
budget exceeds the stored token count.

The within-artifact property does not prove disjointness across separately
materialized artifacts. If a future independent evaluation artifact is added,
it must be checked against all relevant wiring and training artifacts with
`scripts/verify_data_disjointness.py`.

The disjointness tool compares exact complete BOS-delimited tokenized documents.
It excludes leading and trailing fragments. It does not detect every shared
passage or near-duplicate document. A passing report is not proof of no possible
contamination. Retain the artifact identities and the report's coverage limits.
