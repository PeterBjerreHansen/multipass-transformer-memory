from pathlib import Path
import json

import numpy as np
import pytest

from tiny_mistral_mptt.data.manifest import (
    DATA_FORMAT_VERSION,
    PACKING_POLICY,
    DataManifest,
    file_sha256,
    verify_artifact,
)
from tiny_mistral_mptt.data.disjointness import compare_document_disjointness
from tiny_mistral_mptt.data.packed_dataset import (
    PackedTokenDataset,
    load_packed_dataset_for_experiment,
)
from tiny_mistral_mptt.data.prepare import PreparationRequest, materialize_from_document_iterators
from tiny_mistral_mptt.data.recipes import DOLMINO_50B_SOURCES, allocate_blocks


def make_iter(source_index: int):
    for doc in range(10_000):
        # Source/doc identity is embedded in the fake text so consuming validation
        # before training visibly advances the source stream.
        yield f"source{source_index}-document{doc}-abcdefghijklmnopqrstuvwxyz"


def fake_tokenize(text: str) -> list[int]:
    return [3 + (ord(char) % 60) for char in text]


def materialize(
    root: Path,
    *,
    train_skip_tokens: int = 0,
    validation_skip_tokens: int = 0,
):
    request = PreparationRequest(
        output_dir=root,
        sequence_length=16,
        train_tokens=16 * 120,
        validation_tokens=16 * 60,
        seed=123,
        dataset_repo="fake/dolmino",
        requested_revision="test",
        resolved_revision="deadbeef",
        tokenizer_file=Path("tokenizer.json"),
        tokenizer_sha256="abc123",
        vocab_size=97,
        bos_token_id=1,
        forbidden_token_ids=(96,),
        train_skip_tokens=train_skip_tokens,
        validation_skip_tokens=validation_skip_tokens,
    )
    return materialize_from_document_iterators(
        request,
        iterators={source.name: iter(make_iter(i)) for i, source in enumerate(DOLMINO_50B_SOURCES)},
        tokenize=fake_tokenize,
    )


def test_materialization_is_deterministic_and_source_exact(tmp_path):
    first = materialize(tmp_path / "a")
    second = materialize(tmp_path / "b")
    assert first.train.data_sha256 == second.train.data_sha256
    assert first.validation.data_sha256 == second.validation.data_sha256
    assert first.train.source_sha256 == second.train.source_sha256
    assert first.train.blocks_by_source == allocate_blocks(120)
    assert first.validation.blocks_by_source == allocate_blocks(60)
    assert first.format_version == DATA_FORMAT_VERSION
    assert first.packing_policy == PACKING_POLICY
    assert first.forbidden_token_ids == (96,)
    assert DataManifest.read(tmp_path / "a" / "manifest.json") == first


def test_packed_dataset_reads_fixed_unpadded_blocks(tmp_path):
    manifest = materialize(tmp_path / "artifact")
    train = PackedTokenDataset(tmp_path / "artifact", "train")
    val = PackedTokenDataset(tmp_path / "artifact", "validation")
    assert len(train) == manifest.train.blocks
    assert train.block(0).shape == (16,)
    assert train.batch([0, 1, 2]).shape == (3, 16)
    assert 0 <= train.source_id(0) < len(DOLMINO_50B_SOURCES)
    # Validation is consumed from the source iterators before training. The two
    # deterministic packed artifacts therefore do not start with the same data.
    assert not train.block(0).equal(val.block(0))


def test_materialization_rejects_forbidden_control_tokens(tmp_path):
    request = PreparationRequest(
        output_dir=tmp_path / "forbidden",
        sequence_length=16,
        train_tokens=16 * 120,
        validation_tokens=16 * 60,
        seed=123,
        dataset_repo="fake/dolmino",
        requested_revision="test",
        resolved_revision="deadbeef",
        tokenizer_file=Path("tokenizer.json"),
        tokenizer_sha256="abc123",
        vocab_size=97,
        bos_token_id=1,
        forbidden_token_ids=(7,),
    )
    with pytest.raises(ValueError, match="forbidden control token"):
        materialize_from_document_iterators(
            request,
            iterators={
                source.name: iter(make_iter(i))
                for i, source in enumerate(DOLMINO_50B_SOURCES)
            },
            tokenize=lambda _text: [7],
        )


def test_training_skip_preserves_validation_and_advances_training_stream(tmp_path):
    base = materialize(tmp_path / "base")
    skipped = materialize(tmp_path / "skipped", train_skip_tokens=16 * 60)

    assert skipped.train_skip_tokens == 16 * 60
    assert skipped.validation.data_sha256 == base.validation.data_sha256
    assert skipped.train.data_sha256 != base.train.data_sha256


def test_validation_skip_advances_the_evaluation_stream(tmp_path):
    base = materialize(tmp_path / "base-validation")
    skipped = materialize(
        tmp_path / "skipped-validation",
        validation_skip_tokens=16 * 60,
    )

    assert skipped.validation_skip_tokens == 16 * 60
    assert skipped.validation.data_sha256 != base.validation.data_sha256


def test_document_disjointness_detects_shared_validation_and_clean_train(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    materialize(first_root)
    materialize(second_root)

    shared = compare_document_disjointness(
        reference_dir=first_root,
        reference_split="validation",
        against_dir=second_root,
        against_split="validation",
    )
    assert shared["disjoint"] is False
    assert shared["shared_unique_document_hashes"] > 0

    clean = compare_document_disjointness(
        reference_dir=first_root,
        reference_split="validation",
        against_dir=first_root,
        against_split="train",
    )
    assert clean["disjoint"] is True


def test_verify_artifact_detects_mutation(tmp_path):
    root = tmp_path / "artifact-verify"
    materialize(root)
    verify_artifact(root)
    path = root / "validation.sources.bin"
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        verify_artifact(root)


def test_verify_artifact_rejects_malformed_token_file_size(tmp_path):
    root = tmp_path / "artifact-size"
    materialize(root)
    manifest = DataManifest.read(root / "manifest.json")
    train_path = root / manifest.train.data_file
    with train_path.open("r+b") as handle:
        handle.truncate(train_path.stat().st_size - 1)

    with pytest.raises(ValueError, match="size mismatch"):
        verify_artifact(root)


def test_packed_dataset_integrity_option_rejects_mutation(tmp_path):
    root = tmp_path / "artifact-integrity"
    materialize(root)
    path = root / "validation.bin"
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum"):
        load_packed_dataset_for_experiment(
            root,
            "validation",
            verify_integrity=True,
        )


def test_verify_artifact_rejects_forbidden_token_even_with_matching_checksum(tmp_path):
    root = tmp_path / "artifact-forbidden-token"
    materialize(root)
    manifest_path = root / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = root / raw["train"]["data_file"]
    tokens = np.memmap(train_path, mode="r+", dtype=np.uint16)
    tokens[0] = 96
    tokens.flush()
    del tokens
    raw["train"]["data_sha256"] = file_sha256(train_path)
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden control token"):
        verify_artifact(root)


def test_packed_dataset_rejects_legacy_artifact_before_loading(tmp_path):
    root = tmp_path / "legacy-artifact"
    materialize(root)
    manifest_path = root / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["format_version"] = DATA_FORMAT_VERSION - 1
    raw.pop("packing_policy")
    raw.pop("forbidden_token_ids")
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported data artifact format"):
        PackedTokenDataset(root, "train")


def test_verify_artifact_rejects_invalid_training_offset(tmp_path):
    root = tmp_path / "artifact-offset"
    materialize(root)
    manifest_path = root / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["train_skip_tokens"] = 1
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="split-stream offset"):
        verify_artifact(root)
