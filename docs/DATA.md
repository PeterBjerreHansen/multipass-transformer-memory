# Dolmino data artifact

Training does not live-stream remote data. A one-time materialization converts a
pinned slice of `allenai/dolmino-mix-1124` into deterministic local token blocks.
Every architecture therefore starts from the same linguistic IDs and block
order.

Checked-in preparation recipes live beside their local generated artifacts:

```text
data/dolmino/wiring_2048/config.yaml
data/dolmino/pilot_2048/config.yaml
data/dolmino/gpu_2048/config.yaml
data/dolmino/gpu_2048_staged/config.yaml
```

Generated binaries and manifests remain local/ignored.

## Split construction

For each source, validation documents are consumed first from a deterministic
shuffled streaming iterator; the final partially used validation document is
discarded. Training then continues from the next document. Thus train and
validation are source-document-disjoint **within one materialized artifact**.
An optional `train_skip_tokens` consumes and discards complete source-balanced
blocks after validation and before the stored training split. This makes
purpose-specific artifacts non-overlapping while keeping their shared
validation bytes identical.

Documents are tokenized with the pinned TinyMistral tokenizer, with BOS used as
an explicit document separator. Packed blocks are fixed-length and unpadded.

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
revision, recipe, shuffle settings, seed, training-stream offset, and file
hashes.

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
**linguistic** tokens. At C=8 the bank model processes 2303 physical positions.
That extra compute is intentional and separately accounted; it avoids silently
reducing the text/data dose for MEM experiments.

The model's maximum position range must be large enough for the expanded block;
training preflight checks this.

## Core-run split ownership

The document-disjoint guarantee belongs to one materialized artifact.
Development wiring uses `wiring_2048` exactly once per arm. Development pilots
use `pilot_2048` exactly once when continued to their full 10M endpoint. Both
share the same held-out validation split; the pilot recipe skips the complete
5M wiring slice before storing its following 10M training slice. The larger
`gpu_2048` artifact supports ordinary serious runs. The `gpu_2048_staged`
recipe stores non-overlapping wiring and Phase-B slices with a shared validation
set for initialized controls. A run may reshuffle and repeat its assigned slice
when its budget exceeds the stored token count. Evaluation remains
source-document-disjoint from training in every artifact.
