from __future__ import annotations

from pathlib import Path
from typing import Iterator

from tiny_mistral.config import MistralConfig

from .manifest import file_sha256
from .prepare import PreparationRequest, materialize_from_document_iterators
from .recipes import DOLMINO_50B_SOURCES, DOLMINO_REFERENCE_REVISION, DOLMINO_REPO_ID


def _lazy_dependencies():
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Dolmino preparation requires: uv sync --extra data") from exc
    return load_dataset, HfApi, Tokenizer


def prepare_dolmino(
    *,
    output_dir: str | Path,
    model_dir: str | Path,
    sequence_length: int,
    train_tokens: int,
    validation_tokens: int,
    validation_skip_tokens: int = 0,
    train_skip_tokens: int = 0,
    seed: int = 1337,
    dataset_repo: str = DOLMINO_REPO_ID,
    revision: str = DOLMINO_REFERENCE_REVISION,
    shuffle_buffer: int = 10_000,
):
    load_dataset, HfApi, Tokenizer = _lazy_dependencies()
    model_dir = Path(model_dir)
    tokenizer_path = model_dir / "tokenizer.json"
    config = MistralConfig.from_json_file(model_dir / "config.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    resolved = HfApi().dataset_info(dataset_repo, revision=revision).sha

    iterators: dict[str, Iterator[str]] = {}
    for index, source in enumerate(DOLMINO_50B_SOURCES):
        dataset = load_dataset(
            dataset_repo,
            source.config_name,
            split="train",
            streaming=True,
            revision=resolved,
        )
        dataset = dataset.shuffle(seed=seed + 10_007 * index, buffer_size=shuffle_buffer)
        rows = iter(dataset)

        def text_iterator(rows=rows):
            for row in rows:
                text = row.get("text")
                if isinstance(text, str) and text:
                    yield text

        iterators[source.name] = iter(text_iterator())

    request = PreparationRequest(
        output_dir=Path(output_dir),
        sequence_length=sequence_length,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        validation_skip_tokens=validation_skip_tokens,
        train_skip_tokens=train_skip_tokens,
        seed=seed,
        dataset_repo=dataset_repo,
        requested_revision=revision,
        resolved_revision=resolved,
        tokenizer_file=tokenizer_path,
        tokenizer_sha256=file_sha256(tokenizer_path),
        vocab_size=config.vocab_size,
        bos_token_id=config.bos_token_id,
        recipe_name="dolmino_50b",
        shuffle_buffer=shuffle_buffer,
    )
    return materialize_from_document_iterators(
        request,
        iterators=iterators,
        tokenize=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
    )
