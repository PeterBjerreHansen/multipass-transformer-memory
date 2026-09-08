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


def load_packed_dataset_for_experiment(
    artifact_dir: str | Path,
    split: str,
    *,
    verify_integrity: bool = False,
) -> PackedTokenDataset:
    artifact_dir = Path(artifact_dir)
    if verify_integrity:
        verify_artifact(artifact_dir)
    return PackedTokenDataset(artifact_dir, split)
