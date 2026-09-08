from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from .manifest import DataManifest, validate_manifest_contract, verify_artifact


class PackedTokenDataset:
    """Memory-mapped, unpadded fixed-length token blocks plus a source id per block."""

    def __init__(self, artifact_dir: str | Path, split: str):
        self.artifact_dir = Path(artifact_dir)
        self.manifest = DataManifest.read(self.artifact_dir / "manifest.json")
        validate_manifest_contract(self.manifest)
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.split = split
        info = getattr(self.manifest, split)
        self.sequence_length = self.manifest.sequence_length
        data_path = self.artifact_dir / info.data_file
        source_path = self.artifact_dir / info.source_file
        expected_data_bytes = info.blocks * self.sequence_length * np.dtype(np.uint16).itemsize
        expected_source_bytes = info.blocks * np.dtype(np.uint8).itemsize
        if data_path.stat().st_size != expected_data_bytes:
            raise ValueError(f"unexpected packed token file size: {data_path}")
        if source_path.stat().st_size != expected_source_bytes:
            raise ValueError(f"unexpected source-id file size: {source_path}")
        self._tokens = np.memmap(
            data_path,
            mode="r",
            dtype=np.uint16,
            shape=(info.blocks, self.sequence_length),
        )
        self._sources = np.memmap(source_path, mode="r", dtype=np.uint8, shape=(info.blocks,))

    def __len__(self) -> int:
        return int(self._tokens.shape[0])

    def block(self, index: int, *, device: torch.device | str | None = None) -> torch.Tensor:
        # Copy avoids exposing a read-only mmap through torch.from_numpy.
        array = np.array(self._tokens[index], dtype=np.int64, copy=True)
        return torch.tensor(array, dtype=torch.long, device=device)

    def batch(self, indices: list[int], *, device: torch.device | str | None = None) -> torch.Tensor:
        array = np.array(self._tokens[indices], dtype=np.int64, copy=True)
        return torch.tensor(array, dtype=torch.long, device=device)

    def source_id(self, index: int) -> int:
        return int(self._sources[index])

    def source_ids(self, indices: list[int]) -> list[int]:
        return [int(x) for x in self._sources[indices]]


class StatefulBlockSampler:
    """Finite shuffled epochs with a serializable RNG/order/position."""

    def __init__(self, size: int, *, seed: int):
        if size <= 0:
            raise ValueError("sampler size must be positive")
        self.size = int(size)
        self.rng = random.Random(seed)
        self.order = list(range(size))
        self.rng.shuffle(self.order)
        self.position = 0
        self.epoch = 0

    def _reshuffle(self) -> None:
        self.order = list(range(self.size))
        self.rng.shuffle(self.order)
        self.position = 0
        self.epoch += 1

    def next_indices(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("count must be positive")
        result: list[int] = []
        while len(result) < count:
            if self.position >= self.size:
                self._reshuffle()
            take = min(count - len(result), self.size - self.position)
            result.extend(self.order[self.position : self.position + take])
            self.position += take
        return result

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "rng_state": self.rng.getstate(),
            "order": list(self.order),
            "position": self.position,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["size"]) != self.size:
            raise ValueError("sampler size changed across resume")
        self.rng.setstate(state["rng_state"])
        self.order = [int(x) for x in state["order"]]
        self.position = int(state["position"])
        self.epoch = int(state["epoch"])
        if sorted(self.order) != list(range(self.size)):
            raise ValueError("invalid serialized sampler permutation")
        if not 0 <= self.position <= self.size:
            raise ValueError("invalid serialized sampler position")


def memory_token_physical_length(linguistic_length: int, interval: int) -> int:
    """Physical positions after inserting a MEM slot every ``interval`` data tokens.

    No trailing MEM is inserted after the final linguistic token because it
    would have no future token to serve inside the packed block.
    """
    if linguistic_length < 1 or interval < 1:
        raise ValueError("linguistic_length and interval must be positive")
    return linguistic_length + (linguistic_length - 1) // interval


def insert_memory_tokens(
    ids: torch.Tensor,
    *,
    memory_token_id: int,
    interval: int,
) -> torch.Tensor:
    """Insert input-only memory-control positions into ordinary token blocks.

    ``ids`` is [B,T] and must contain only linguistic vocabulary IDs. A MEM is
    inserted after each complete group of ``interval`` linguistic tokens when
    at least one later linguistic token remains. The transformation is
    deterministic and preserves the ordinary-token order exactly.
    """
    if ids.ndim != 2:
        raise ValueError("ids must be [B,T]")
    if interval < 1 or memory_token_id < 0:
        raise ValueError("interval must be positive and memory_token_id non-negative")
    seq_len = ids.shape[1]
    pieces: list[torch.Tensor] = []
    cursor = 0
    while cursor < seq_len:
        end = min(cursor + interval, seq_len)
        pieces.append(ids[:, cursor:end])
        cursor = end
        if cursor < seq_len:
            pieces.append(
                torch.full(
                    (ids.shape[0], 1),
                    int(memory_token_id),
                    dtype=ids.dtype,
                    device=ids.device,
                )
            )
    return torch.cat(pieces, dim=1)


class MemoryTokenPackedDataset:
    """Deterministic view inserting ``<MEM>`` into an ordinary packed artifact.

    The backing artifact remains linguistically tokenized and source/provenance
    stable.  The view adds architecture control positions only at load time,
    which keeps ``max_unique_tokens`` interpretable as actual data tokens.
    Different memory cadences should use backing artifacts whose linguistic
    sequence lengths expand to the same desired physical context length.
    """

    def __init__(self, base: PackedTokenDataset, *, interval: int):
        if interval < 1:
            raise ValueError("memory-token interval must be positive")
        self.base = base
        self.interval = int(interval)
        self.memory_token_id = int(base.manifest.vocab_size)
        self.manifest = base.manifest
        self.artifact_dir = base.artifact_dir
        self.split = base.split
        self.linguistic_sequence_length = int(base.sequence_length)
        self.sequence_length = memory_token_physical_length(
            self.linguistic_sequence_length, self.interval
        )

    def __len__(self) -> int:
        return len(self.base)

    def block(self, index: int, *, device: torch.device | str | None = None) -> torch.Tensor:
        ordinary = self.base.block(index, device=device)[None, :]
        return insert_memory_tokens(
            ordinary, memory_token_id=self.memory_token_id, interval=self.interval
        )[0]

    def batch(self, indices: list[int], *, device: torch.device | str | None = None) -> torch.Tensor:
        ordinary = self.base.batch(indices, device=device)
        return insert_memory_tokens(
            ordinary, memory_token_id=self.memory_token_id, interval=self.interval
        )

    def source_id(self, index: int) -> int:
        return self.base.source_id(index)

    def source_ids(self, indices: list[int]) -> list[int]:
        return self.base.source_ids(indices)


def load_packed_dataset_for_experiment(
    artifact_dir: str | Path,
    split: str,
    *,
    memory_write_mode: str | None = None,
    memory_write_stride: int | None = None,
    verify_integrity: bool = False,
) -> PackedTokenDataset | MemoryTokenPackedDataset:
    artifact_dir = Path(artifact_dir)
    if verify_integrity:
        verify_artifact(artifact_dir)
    base = PackedTokenDataset(artifact_dir, split)
    if memory_write_mode in {"memory_token"}:
        if memory_write_stride is None:
            raise ValueError("memory_token dataset view requires memory_write_stride")
        return MemoryTokenPackedDataset(base, interval=int(memory_write_stride))
    return base
