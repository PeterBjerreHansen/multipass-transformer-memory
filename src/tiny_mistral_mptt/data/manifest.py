from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def read(cls, path: str | Path) -> "DataManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["train"] = PackedSplitInfo(**raw["train"])
        raw["validation"] = PackedSplitInfo(**raw["validation"])
        return cls(**raw)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(artifact_dir: str | Path) -> DataManifest:
    artifact_dir = Path(artifact_dir)
    manifest = DataManifest.read(artifact_dir / "manifest.json")
    if manifest.format_version != 1:
        raise ValueError("unsupported data artifact format")
    if (
        manifest.validation_skip_tokens < 0
        or manifest.validation_skip_tokens % manifest.sequence_length
        or manifest.train_skip_tokens < 0
        or manifest.train_skip_tokens % manifest.sequence_length
    ):
        raise ValueError("invalid split-stream offset in data manifest")
    for split_name in ("train", "validation"):
        info = getattr(manifest, split_name)
        data_path = artifact_dir / info.data_file
        source_path = artifact_dir / info.source_file
        if not data_path.is_file() or not source_path.is_file():
            raise FileNotFoundError(f"missing {split_name} artifact files")
        if file_sha256(data_path) != info.data_sha256:
            raise ValueError(f"{split_name} token artifact checksum mismatch")
        if file_sha256(source_path) != info.source_sha256:
            raise ValueError(f"{split_name} source-id artifact checksum mismatch")
    return manifest
