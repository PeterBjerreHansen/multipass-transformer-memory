from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import random
import shutil

import numpy as np

from .manifest import (
    DATA_FORMAT_VERSION,
    PACKING_POLICY,
    DataManifest,
    PackedSplitInfo,
    file_sha256,
)
from .recipes import DOLMINO_50B_SOURCES, allocate_blocks, normalized_weights


TokenizerFn = Callable[[str], list[int]]


@dataclass(frozen=True)
class PreparationRequest:
    output_dir: Path
    sequence_length: int
    train_tokens: int
    validation_tokens: int
    seed: int
    dataset_repo: str
    requested_revision: str
    resolved_revision: str
    tokenizer_file: Path
    tokenizer_sha256: str
    vocab_size: int
    bos_token_id: int
    forbidden_token_ids: tuple[int, ...]
    recipe_name: str = "dolmino_50b"
    shuffle_buffer: int | None = None
    train_skip_tokens: int = 0
    validation_skip_tokens: int = 0

    def validate(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.train_tokens < self.sequence_length or self.validation_tokens < self.sequence_length:
            raise ValueError("each split must request at least one complete block")
        if self.train_tokens % self.sequence_length or self.validation_tokens % self.sequence_length:
            raise ValueError("split token budgets must be exact multiples of sequence_length")
        if (
            self.train_skip_tokens < 0
            or self.train_skip_tokens % self.sequence_length
            or self.validation_skip_tokens < 0
            or self.validation_skip_tokens % self.sequence_length
        ):
            raise ValueError(
                "split skip tokens must be non-negative and divisible by sequence_length"
            )
        if self.vocab_size > np.iinfo(np.uint16).max + 1:
            raise ValueError("vocab_size does not fit uint16 artifact format")
        if not 0 <= self.bos_token_id < self.vocab_size:
            raise ValueError("invalid BOS token id")
        if any(not 0 <= token < self.vocab_size for token in self.forbidden_token_ids):
            raise ValueError("forbidden token id is outside the declared vocabulary")
        if not self.forbidden_token_ids:
            raise ValueError("at least one forbidden control-token id must be declared")
        if len(set(self.forbidden_token_ids)) != len(self.forbidden_token_ids):
            raise ValueError("forbidden control-token ids must be unique")
        if self.bos_token_id in self.forbidden_token_ids:
            raise ValueError("BOS cannot also be a forbidden control token")


def _write_source_blocks(
    documents: Iterator[str],
    *,
    output_path: Path | None,
    blocks: int,
    sequence_length: int,
    bos_token_id: int,
    tokenize: TokenizerFn,
    vocab_size: int,
    forbidden_token_ids: tuple[int, ...] = (),
) -> Path | None:
    """Consume one source quota, optionally writing its packed blocks."""
    if blocks <= 0:
        raise ValueError("every published source must receive at least one block")
    target = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        target = np.memmap(
            output_path,
            mode="w+",
            dtype=np.uint16,
            shape=(blocks, sequence_length),
        )
    row = 0
    buffer: list[int] = []
    cursor = 0
    while row < blocks:
        try:
            text = next(documents)
        except StopIteration as exc:
            raise RuntimeError("source exhausted before its requested token quota") from exc
        if not isinstance(text, str) or not text:
            continue
        ids = tokenize(text)
        if not ids:
            continue
        if any(token < 0 or token >= vocab_size for token in ids):
            raise ValueError("tokenizer emitted an id outside the declared vocabulary")
        if any(token in forbidden_token_ids for token in ids):
            raise ValueError(
                "tokenizer emitted a forbidden control token; disable tokenizer padding"
            )
        # Explicit document separator. If the quota is reached inside this
        # document, its unused suffix is intentionally discarded. The next
        # split therefore starts from the following document.
        buffer.append(bos_token_id)
        buffer.extend(ids)
        while len(buffer) - cursor >= sequence_length and row < blocks:
            if target is not None:
                target[row] = buffer[cursor : cursor + sequence_length]
            cursor += sequence_length
            row += 1
        # Keep only an incomplete tail between documents. This caps the buffer
        # at roughly one document plus one block instead of the whole quota.
        if cursor:
            buffer = buffer[cursor:]
            cursor = 0
    if target is not None:
        target.flush()
        del target
    return output_path


def _write_split(
    output_dir: Path,
    split: str,
    source_files: dict[str, Path],
    counts: dict[str, int],
    *,
    sequence_length: int,
    seed: int,
    source_ids: dict[str, int],
) -> PackedSplitInfo:
    schedule: list[str] = []
    for name, count in counts.items():
        schedule.extend([name] * count)
    rng = random.Random(seed)
    rng.shuffle(schedule)
    data_path = output_dir / f"{split}.bin"
    source_path = output_dir / f"{split}.sources.bin"
    cursor = {name: 0 for name in source_files}
    sources = {
        name: np.memmap(path, mode="r", dtype=np.uint16, shape=(counts[name], sequence_length))
        for name, path in source_files.items()
    }
    data_mm = np.memmap(data_path, mode="w+", dtype=np.uint16, shape=(len(schedule), sequence_length))
    source_mm = np.memmap(source_path, mode="w+", dtype=np.uint8, shape=(len(schedule),))
    for row, name in enumerate(schedule):
        index = cursor[name]
        data_mm[row] = sources[name][index]
        source_mm[row] = source_ids[name]
        cursor[name] += 1
    data_mm.flush(); source_mm.flush()
    del data_mm, source_mm
    sources.clear()
    return PackedSplitInfo(
        blocks=len(schedule),
        tokens=len(schedule) * sequence_length,
        data_file=data_path.name,
        source_file=source_path.name,
        data_sha256=file_sha256(data_path),
        source_sha256=file_sha256(source_path),
        blocks_by_source=dict(counts),
    )


def materialize_from_document_iterators(
    request: PreparationRequest,
    *,
    iterators: dict[str, Iterator[str]],
    tokenize: TokenizerFn,
) -> DataManifest:
    """Core deterministic materializer, dependency-free except NumPy."""
    request.validate()
    source_names = [item.name for item in DOLMINO_50B_SOURCES]
    missing = sorted(set(source_names) - set(iterators))
    if missing:
        raise ValueError(f"missing document iterators for sources: {missing}")
    train_blocks_total = request.train_tokens // request.sequence_length
    val_blocks_total = request.validation_tokens // request.sequence_length
    skip_blocks_total = request.train_skip_tokens // request.sequence_length
    validation_skip_blocks_total = (
        request.validation_skip_tokens // request.sequence_length
    )
    if train_blocks_total < len(source_names) or val_blocks_total < len(source_names):
        raise ValueError(
            "requested splits are too small to represent every Dolmino source; "
            f"need at least {len(source_names) * request.sequence_length} tokens per split"
        )
    if 0 < skip_blocks_total < len(source_names):
        raise ValueError(
            "a non-zero training skip must cover at least one block per Dolmino source"
        )
    if 0 < validation_skip_blocks_total < len(source_names):
        raise ValueError(
            "a non-zero validation skip must cover at least one block per Dolmino source"
        )
    train_alloc = allocate_blocks(train_blocks_total)
    val_alloc = allocate_blocks(val_blocks_total)
    skip_alloc = allocate_blocks(skip_blocks_total) if skip_blocks_total else None
    validation_skip_alloc = (
        allocate_blocks(validation_skip_blocks_total)
        if validation_skip_blocks_total
        else None
    )
    source_ids = {name: index for index, name in enumerate(source_names)}

    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir / ".prepare_tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    try:
        val_files: dict[str, Path] = {}
        train_files: dict[str, Path] = {}
        for name in source_names:
            if validation_skip_alloc is not None:
                _write_source_blocks(
                    iterators[name],
                    output_path=None,
                    blocks=validation_skip_alloc[name],
                    sequence_length=request.sequence_length,
                    bos_token_id=request.bos_token_id,
                    tokenize=tokenize,
                    vocab_size=request.vocab_size,
                    forbidden_token_ids=request.forbidden_token_ids,
                )
            # Consume validation first from each persistent shuffled iterator.
            validation_path = _write_source_blocks(
                iterators[name],
                output_path=temporary_dir / f"validation.{name}.bin",
                blocks=val_alloc[name],
                sequence_length=request.sequence_length,
                bos_token_id=request.bos_token_id,
                tokenize=tokenize,
                vocab_size=request.vocab_size,
                forbidden_token_ids=request.forbidden_token_ids,
            )
            assert validation_path is not None
            val_files[name] = validation_path
            if skip_alloc is not None:
                _write_source_blocks(
                    iterators[name],
                    output_path=None,
                    blocks=skip_alloc[name],
                    sequence_length=request.sequence_length,
                    bos_token_id=request.bos_token_id,
                    tokenize=tokenize,
                    vocab_size=request.vocab_size,
                    forbidden_token_ids=request.forbidden_token_ids,
                )
            train_path = _write_source_blocks(
                iterators[name],
                output_path=temporary_dir / f"train.{name}.bin",
                blocks=train_alloc[name],
                sequence_length=request.sequence_length,
                bos_token_id=request.bos_token_id,
                tokenize=tokenize,
                vocab_size=request.vocab_size,
                forbidden_token_ids=request.forbidden_token_ids,
            )
            assert train_path is not None
            train_files[name] = train_path

        validation = _write_split(
            output_dir, "validation", val_files, val_alloc,
            sequence_length=request.sequence_length,
            seed=request.seed ^ 0xA5A5,
            source_ids=source_ids,
        )
        train = _write_split(
            output_dir, "train", train_files, train_alloc,
            sequence_length=request.sequence_length,
            seed=request.seed ^ 0x5A5A,
            source_ids=source_ids,
        )
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    weights = normalized_weights()
    manifest = DataManifest(
        format_version=DATA_FORMAT_VERSION,
        dataset_repo=request.dataset_repo,
        requested_revision=request.requested_revision,
        resolved_revision=request.resolved_revision,
        tokenizer_file=str(request.tokenizer_file),
        tokenizer_sha256=request.tokenizer_sha256,
        vocab_size=request.vocab_size,
        bos_token_id=request.bos_token_id,
        forbidden_token_ids=request.forbidden_token_ids,
        sequence_length=request.sequence_length,
        preparation_seed=request.seed,
        recipe_name=request.recipe_name,
        shuffle_buffer=request.shuffle_buffer,
        train_skip_tokens=request.train_skip_tokens,
        validation_skip_tokens=request.validation_skip_tokens,
        packing_policy=PACKING_POLICY,
        source_ids=source_ids,
        mixture_weights={source.name: weight for source, weight in zip(DOLMINO_50B_SOURCES, weights)},
        train=train,
        validation=validation,
    )
    manifest.write(output_dir / "manifest.json")
    return manifest
