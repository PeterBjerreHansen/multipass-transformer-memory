from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .recipes import DOLMINO_REFERENCE_REVISION, DOLMINO_REPO_ID


@dataclass(slots=True)
class DataPreparationConfig:
    output_dir: str = "data/dolmino/wiring_2048"
    model_dir: str = "checkpoints/TinyMistral-248M-v3"
    sequence_length: int = 2048
    train_tokens: int = 5_242_880
    validation_tokens: int = 524_288
    validation_skip_tokens: int = 0
    train_skip_tokens: int = 0
    seed: int = 1337
    dataset_repo: str = DOLMINO_REPO_ID
    revision: str = DOLMINO_REFERENCE_REVISION
    shuffle_buffer: int = 25_000

    def validate(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.train_tokens <= 0 or self.validation_tokens <= 0:
            raise ValueError("token budgets must be positive")
        if self.validation_skip_tokens < 0 or self.train_skip_tokens < 0:
            raise ValueError("split skip tokens must be non-negative")
        if (
            self.train_tokens % self.sequence_length
            or self.validation_tokens % self.sequence_length
            or self.validation_skip_tokens % self.sequence_length
            or self.train_skip_tokens % self.sequence_length
        ):
            raise ValueError("token budgets must be exact multiples of sequence_length")
        if self.shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")


def load_data_config(path: str | Path) -> DataPreparationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("data config must be a YAML mapping")
    known = set(DataPreparationConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown data config fields: {unknown}")
    cfg = DataPreparationConfig(**raw)
    cfg.validate()
    return cfg
