#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiny_mistral_mptt.data.disjointness import compare_document_disjointness
from tiny_mistral_mptt.data.manifest import verify_artifact


def _artifact_split(value: str) -> tuple[Path, str]:
    try:
        artifact, split = value.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected ARTIFACT:train|validation") from exc
    if split not in {"train", "validation"}:
        raise argparse.ArgumentTypeError("split must be train or validation")
    return Path(artifact), split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject BOS-delimited document overlap between packed artifacts."
    )
    parser.add_argument("--reference", required=True, type=_artifact_split)
    parser.add_argument("--against", required=True, action="append", type=_artifact_split)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="skip full artifact checksums when they were verified separately",
    )
    args = parser.parse_args()

    reference_dir, reference_split = args.reference
    artifacts = {reference_dir, *(artifact for artifact, _ in args.against)}
    if not args.skip_checksums:
        for artifact in artifacts:
            verify_artifact(artifact)

    comparisons = [
        compare_document_disjointness(
            reference_dir=reference_dir,
            reference_split=reference_split,
            against_dir=against_dir,
            against_split=against_split,
        )
        for against_dir, against_split in args.against
    ]
    document = {
        "method": "complete_bos_delimited_tokenized_document_sha256",
        "disjoint": all(item["disjoint"] for item in comparisons),
        "comparisons": comparisons,
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not document["disjoint"]:
        raise SystemExit("FAIL: evaluation documents overlap a compared artifact")
    print("PASS: no complete tokenized documents overlap")


if __name__ == "__main__":
    main()
