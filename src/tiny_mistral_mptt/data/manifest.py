from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DATA_FORMAT_VERSION = 2
PACKING_POLICY = "raw_unpadded_document_stream_v1"


@dataclass(frozen=True)
class PackedSplitInfo:
    blocks: int
    tokens: int
    data_file: str
    source_file: str
    data_sha256: str
    source_sha256: str
    blocks_by_source: dict[str, int]


@dataclass(frozen=True)
class DataManifest:
    format_version: int
    dataset_repo: str
    requested_revision: str
    resolved_revision: str
    tokenizer_file: str
    tokenizer_sha256: str
    vocab_size: int
    bos_token_id: int
    forbidden_token_ids: tuple[int, ...]
    sequence_length: int
    preparation_seed: int
    recipe_name: str
    shuffle_buffer: int | None
    source_ids: dict[str, int]
    mixture_weights: dict[str, float]
    train: PackedSplitInfo
    validation: PackedSplitInfo
    train_skip_tokens: int = 0
    validation_skip_tokens: int = 0
    packing_policy: str = "legacy_unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def read(cls, path: str | Path) -> "DataManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # Retain a clear unsupported-format error for legacy manifests, which
        # predate the recorded control-token audit.
        raw["forbidden_token_ids"] = tuple(raw.get("forbidden_token_ids", ()))
        raw["train"] = PackedSplitInfo(**raw["train"])
        raw["validation"] = PackedSplitInfo(**raw["validation"])
        return cls(**raw)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_contract(manifest: DataManifest) -> None:
    """Reject artifacts that do not satisfy the current packing contract."""
    if manifest.format_version != DATA_FORMAT_VERSION:
        raise ValueError("unsupported data artifact format")
    if manifest.packing_policy != PACKING_POLICY:
        raise ValueError("data artifact was not prepared with the raw unpadded packing policy")
    if not manifest.forbidden_token_ids:
        raise ValueError("data manifest does not record any forbidden control-token ids")
    if len(set(manifest.forbidden_token_ids)) != len(manifest.forbidden_token_ids):
        raise ValueError("data manifest contains duplicate forbidden control-token ids")
    if any(
        token < 0 or token >= manifest.vocab_size
        for token in manifest.forbidden_token_ids
    ):
        raise ValueError("data manifest contains an invalid forbidden control-token id")
    if manifest.bos_token_id in manifest.forbidden_token_ids:
        raise ValueError("BOS cannot also be a forbidden control token")
    if (
        manifest.validation_skip_tokens < 0
        or manifest.validation_skip_tokens % manifest.sequence_length
        or manifest.train_skip_tokens < 0
        or manifest.train_skip_tokens % manifest.sequence_length
    ):
        raise ValueError("invalid split-stream offset in data manifest")


def _verify_token_file(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    forbidden_token_ids: tuple[int, ...],
) -> None:
    """Check size and checksum while rejecting forbidden IDs in one pass."""
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            "packed token artifact size mismatch: "
            f"{path} (expected {expected_bytes} bytes, got {actual_bytes})"
        )
    digest = hashlib.sha256()
    forbidden = np.asarray(forbidden_token_ids, dtype=np.uint16)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            tokens = np.frombuffer(chunk, dtype=np.uint16)
            if np.isin(tokens, forbidden).any():
                raise ValueError(f"packed token artifact contains a forbidden control token: {path}")
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"packed token artifact checksum mismatch: {path}")


def verify_artifact(artifact_dir: str | Path) -> DataManifest:
    artifact_dir = Path(artifact_dir)
    manifest = DataManifest.read(artifact_dir / "manifest.json")
    validate_manifest_contract(manifest)
    for split_name in ("train", "validation"):
        info = getattr(manifest, split_name)
        data_path = artifact_dir / info.data_file
        source_path = artifact_dir / info.source_file
        if not data_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"missing {split_name} artifact files")
        _verify_token_file(
            data_path,
            expected_bytes=info.blocks
            * manifest.sequence_length
            * np.dtype(np.uint16).itemsize,
            expected_sha256=info.data_sha256,
            forbidden_token_ids=manifest.forbidden_token_ids,
        )
        if file_sha256(source_path) != info.source_sha256:
            raise ValueError(f"{split_name} source-id artifact checksum mismatch")
    return manifest
