#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from tiny_mistral_mptt.data.manifest import verify_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a prepared Dolmino artifact against manifest checksums.")
    parser.add_argument("data_dir")
    args = parser.parse_args()
    manifest = verify_artifact(args.data_dir)
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    print("PASS: prepared data artifact integrity and forbidden-token checks passed")


if __name__ == "__main__":
    main()
