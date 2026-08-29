from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import DataManifest, file_sha256


@dataclass(frozen=True)
class DocumentFingerprint:
    source: str
    document_index: int
    tokens: int


def _packed_arrays(
    artifact_dir: Path, split: str
) -> tuple[DataManifest, np.memmap, np.memmap]:
    manifest = DataManifest.read(artifact_dir / "manifest.json")
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    info = getattr(manifest, split)
    tokens = np.memmap(
        artifact_dir / info.data_file,
        mode="r",
        dtype=np.uint16,
        shape=(info.blocks, manifest.sequence_length),
    )
    sources = np.memmap(
        artifact_dir / info.source_file,
        mode="r",
        dtype=np.uint8,
        shape=(info.blocks,),
    )
    return manifest, tokens, sources


def _scan_document_fingerprints(
    artifact_dir: str | Path,
    split: str,
    *,
    retain_hashes: set[bytes] | None,
) -> tuple[dict[bytes, list[DocumentFingerprint]], int]:
    """Hash complete BOS-delimited documents in one packed split.

    Blocks are globally shuffled, but blocks from each source retain their
    original order. Reassembling each source stream therefore recovers every
    complete document except an optional leading/trailing boundary fragment.
    """
    root = Path(artifact_dir)
    manifest, tokens, sources = _packed_arrays(root, split)
    id_to_source = {value: key for key, value in manifest.source_ids.items()}
    states: dict[int, tuple[Any, int] | None] = {
        source_id: None for source_id in id_to_source
    }
    document_indices = {source_id: 0 for source_id in id_to_source}
    fingerprints: dict[bytes, list[DocumentFingerprint]] = {}
    complete_documents = 0

    for block_index in range(tokens.shape[0]):
        source_id = int(sources[block_index])
        if source_id not in id_to_source:
            raise ValueError(f"unknown source id {source_id} in {split}")
        block = np.asarray(tokens[block_index])
        bos_positions = np.flatnonzero(block == manifest.bos_token_id)
        cursor = 0
        state = states[source_id]
        for bos_position in bos_positions:
            position = int(bos_position)
            if state is not None:
                digest, count = state
                if position > cursor:
                    segment = block[cursor:position]
                    digest.update(segment.tobytes())
                    count += int(segment.size)
                if count:
                    complete_documents += 1
                    fingerprint = DocumentFingerprint(
                        source=id_to_source[source_id],
                        document_index=document_indices[source_id],
                        tokens=count,
                    )
                    document_hash = digest.digest()
                    if retain_hashes is None or document_hash in retain_hashes:
                        fingerprints.setdefault(document_hash, []).append(fingerprint)
                    document_indices[source_id] += 1
            state = (hashlib.sha256(), 0)
            cursor = position + 1
        if state is not None and cursor < block.size:
            digest, count = state
            segment = block[cursor:]
            digest.update(segment.tobytes())
            state = (digest, count + int(segment.size))
        states[source_id] = state

    return fingerprints, complete_documents


def document_fingerprints(
    artifact_dir: str | Path,
    split: str,
) -> dict[bytes, list[DocumentFingerprint]]:
    fingerprints, _ = _scan_document_fingerprints(
        artifact_dir,
        split,
        retain_hashes=None,
    )
    return fingerprints


def compare_document_disjointness(
    *,
    reference_dir: str | Path,
    reference_split: str,
    against_dir: str | Path,
    against_split: str,
    max_examples: int = 20,
) -> dict[str, Any]:
    reference_root = Path(reference_dir)
    against_root = Path(against_dir)
    reference_manifest = DataManifest.read(reference_root / "manifest.json")
    against_manifest = DataManifest.read(against_root / "manifest.json")
    if reference_manifest.tokenizer_sha256 != against_manifest.tokenizer_sha256:
        raise ValueError("cannot compare artifacts tokenized by different tokenizers")
    reference, reference_document_count = _scan_document_fingerprints(
        reference_root,
        reference_split,
        retain_hashes=None,
    )
    against, against_document_count = _scan_document_fingerprints(
        against_root,
        against_split,
        retain_hashes=set(reference),
    )
    shared = sorted(against)
    reference_complete_tokens = sum(
        item.tokens for items in reference.values() for item in items
    )
    shared_reference_documents = sum(len(reference[digest]) for digest in shared)
    shared_reference_tokens = sum(
        item.tokens for digest in shared for item in reference[digest]
    )
    examples = []
    for digest in shared[:max_examples]:
        examples.append(
            {
                "sha256": digest.hex(),
                "reference": [item.__dict__ for item in reference[digest]],
                "against": [item.__dict__ for item in against[digest]],
            }
        )
    return {
        "reference": {
            "artifact": str(reference_root.resolve()),
            "manifest_sha256": file_sha256(reference_root / "manifest.json"),
            "split": reference_split,
            "complete_documents": reference_document_count,
            "unique_document_hashes": len(reference),
        },
        "against": {
            "artifact": str(against_root.resolve()),
            "manifest_sha256": file_sha256(against_root / "manifest.json"),
            "split": against_split,
            "complete_documents": against_document_count,
            "matching_unique_document_hashes": len(against),
        },
        "shared_unique_document_hashes": len(shared),
        "shared_reference_documents": shared_reference_documents,
        "shared_reference_document_fraction": (
            shared_reference_documents / reference_document_count
            if reference_document_count
            else 0.0
        ),
        "shared_reference_complete_document_tokens": shared_reference_tokens,
        "shared_reference_complete_document_token_fraction": (
            shared_reference_tokens / reference_complete_tokens
            if reference_complete_tokens
            else 0.0
        ),
        "disjoint": not shared,
        "examples": examples,
    }
