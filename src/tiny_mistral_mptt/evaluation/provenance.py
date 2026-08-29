from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import random
from typing import Any

import torch

from ..config import ExperimentConfig
from ..data.manifest import file_sha256
from ..training.checkpoint import (
    load_checkpoint_for_evaluation,
    load_model_weights,
)
from ..training.provenance import hardware_provenance, source_provenance


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    """Require either trained weights or an explicitly labelled time-zero run."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="full .pt checkpoint or .safetensors snapshot")
    group.add_argument(
        "--initialized-baseline",
        action="store_true",
        help="evaluate time-zero initialized weights and label them as such",
    )


def seed_evaluation(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_evaluation_weights(
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    checkpoint: str | None,
    initialized_baseline: bool,
) -> dict[str, Any]:
    """Load and identify the exact weights used by an evaluation."""
    if initialized_baseline:
        if checkpoint is not None:
            raise ValueError("initialized baseline and checkpoint are mutually exclusive")
        initialization_source = None
        if config.init_from is not None:
            init_path = Path(config.init_from)
            if not init_path.is_file():
                raise FileNotFoundError(
                    f"initialized baseline source does not exist: {init_path}"
                )
            initialization_source = {
                "path": str(init_path.resolve()),
                "sha256": file_sha256(init_path),
                "metadata": load_model_weights(
                    init_path,
                    model=model,
                    expected_experiment_config=config.to_dict(),
                ),
            }
        return {
            "kind": "initialized_baseline",
            "checkpoint": None,
            "checkpoint_sha256": None,
            "initialization_source": initialization_source,
        }
    if checkpoint is None:
        raise ValueError("a checkpoint is required unless --initialized-baseline is set")

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"evaluation checkpoint does not exist: {path}")
    if path.suffix == ".safetensors":
        metadata = load_model_weights(
            path,
            model=model,
            expected_experiment_config=config.to_dict(),
        )
    else:
        training_manifest = Path(config.data_dir) / "manifest.json"
        metadata = load_checkpoint_for_evaluation(
            path,
            model=model,
            expected_manifest_sha256=file_sha256(training_manifest),
            expected_experiment_config=config.to_dict(),
        )
    return {
        "kind": "trained_checkpoint",
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "metadata": metadata,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    versions["python"] = platform.python_version()
    for distribution in (
        "torch",
        "numpy",
        "safetensors",
        "tokenizers",
        "datasets",
        "lm_eval",
    ):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def evaluation_provenance(
    *,
    config_path: str | Path,
    config: ExperimentConfig,
    weight_identity: dict[str, Any],
    device: torch.device,
    seeds: dict[str, int],
    evaluation_data_dir: str | Path | None = None,
    suite_path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    normalized_config = config.to_dict()
    result: dict[str, Any] = {
        "config_path": str(config_path.resolve()),
        "config_file_sha256": file_sha256(config_path),
        "normalized_config": normalized_config,
        "normalized_config_sha256": canonical_json_sha256(normalized_config),
        "weights": weight_identity,
        "seeds": {name: int(value) for name, value in seeds.items()},
        "source": source_provenance(REPOSITORY_ROOT),
        "hardware": hardware_provenance(device),
        "package_versions": _package_versions(),
    }
    if evaluation_data_dir is not None:
        data_dir = Path(evaluation_data_dir)
        manifest_path = data_dir / "manifest.json"
        result["evaluation_data"] = {
            "directory": str(data_dir.resolve()),
            "manifest_sha256": file_sha256(manifest_path),
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        }
    if suite_path is not None:
        suite = Path(suite_path)
        result["suite"] = {
            "path": str(suite.resolve()),
            "sha256": file_sha256(suite),
        }
    return result


def render_or_write_json(document: dict[str, Any], output: str | None) -> str:
    rendered = json.dumps(document, indent=2, sort_keys=True, default=str)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    return rendered
