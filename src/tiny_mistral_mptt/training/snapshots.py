"""Weights and their identity commit together in one atomic safetensors file.

The JSON sidecar is a repairable human-readable mirror, not a commit marker.
Old snapshots without embedded metadata still use the legacy loader contract.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open, SafetensorError
from safetensors.torch import save_file

from .durable import atomic_write_json, fsync_directory


METADATA_KEY = "tiny_mistral_mptt_snapshot"


def snapshot_metadata(path: Path) -> dict | None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        encoded = (handle.metadata() or {}).get(METADATA_KEY)
    if encoded is None:
        return None
    metadata = json.loads(encoded)
    if not isinstance(metadata, dict) or metadata.get("snapshot_format_version") != 1:
        raise ValueError("unsupported snapshot metadata format")
    required = {"experiment_config", "unique_tokens_seen", "optimizer_steps",
                "model_positions_seen", "data_manifest_sha256", "requested_thresholds"}
    if not required <= metadata.keys():
        raise ValueError("snapshot embedded metadata is incomplete")
    return metadata


def publish_snapshot(path: Path, *, model, metadata: dict) -> dict:
    """Publish or finish a snapshot at the current checkpointed model state.

    A previously committed snapshot is immutable. Only partial/legacy files at
    this exact recovery point are replaced from the authoritative current model.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = None
    if path.exists():
        try:
            existing = snapshot_metadata(path)
        except SafetensorError:
            pass  # An incomplete file is reconstructible at this checkpoint.
    if existing is not None:
        for key in ("unique_tokens_seen", "optimizer_steps", "model_positions_seen",
                    "data_manifest_sha256", "requested_thresholds"):
            if existing[key] != metadata[key]:
                raise ValueError(f"committed snapshot identity differs: {key}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            current = model.state_dict()
            if set(handle.keys()) != set(current) or any(
                not torch.equal(handle.get_tensor(name), value.detach().cpu())
                for name, value in current.items()
            ):
                raise ValueError("committed snapshot weights differ from recovered checkpoint")
        metadata = existing
    else:
        metadata = {**metadata, "snapshot_format_version": 1}
        tensors = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        temporary = path.with_name(path.name + ".tmp")
        save_file(tensors, str(temporary), metadata={METADATA_KEY: json.dumps(metadata, sort_keys=True)})
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
        fsync_directory(path.parent.parent)
    # Retrying after failure here never has to republish committed weights.
    atomic_write_json(path.with_suffix(".json"), metadata)
    return metadata
