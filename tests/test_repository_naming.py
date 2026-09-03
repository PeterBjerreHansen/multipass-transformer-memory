"""Keep retired vocabulary out of active paths, APIs and executable contracts."""

from pathlib import Path
import re

from tiny_mistral_mptt.config import (
    MEMORY_ATTENTION_VARIANT_ALIASES, SUPPORTED_VARIANTS, canonical_variant_name,
)
from tiny_mistral_mptt import variants


ROOT = Path(__file__).resolve().parents[1]
RETIRED = re.compile(
    r"bank|multiscale|\battention_dense_and_strided\b|\bstrided_attention\b|sparse_swa"
    r"|memory_attention_add_hybrid|memory_attention_recirculation_hybrid"
    r"|recirculation_strided_memory_attention|MemoryAttentionAddHybrid"
    r"|MemoryAttentionRecirculationHybrid|RecirculationStridedMemoryAttention"
    r"|DenseAndStridedMemoryAttentionVariant|SparseSWA|StridedAttentionVariant",
    re.I,
)
EXPLICIT_COMPATIBILITY = {
    "src/tiny_mistral_mptt/compatibility.py",
    "tests/test_legacy_variant_compatibility.py",
    "tests/test_repository_naming.py",
}


def test_active_tree_does_not_reintroduce_retired_names():
    paths = [ROOT / "README.md", ROOT / "CONTEXT.md"]
    for directory in ("src", "scripts", "tests", "docs", "benchmarks", "evaluation"):
        paths.extend((ROOT / directory).rglob("*"))
    failures = []
    for path in paths:
        if not path.is_file() or (path.suffix not in {".py", ".md", ".yaml"} and path.name not in {"start-and-watch", "run-cloud-study"}):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("benchmarks/historical/", "docs/research/")) or relative == "docs/FROZEN_WIRING_GRILL_EXCHANGE.md":
            continue
        if RETIRED.search(relative):
            failures.append(f"retired filename: {relative}")
        if relative in EXPLICIT_COMPATIBILITY:
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if relative == "CONTEXT.md" and line.startswith("_Avoid_:"):
                continue  # The glossary must identify terms that must not be used.
            if RETIRED.search(line):
                failures.append(f"{relative}:{line_number}: {line.strip()}")
    assert not failures, "\n".join(failures)


def test_registry_and_public_exports_use_current_names():
    names = (
        set(SUPPORTED_VARIANTS) | set(MEMORY_ATTENTION_VARIANT_ALIASES)
        | set(MEMORY_ATTENTION_VARIANT_ALIASES.values()) | set(variants.__all__)
        | {canonical_variant_name(name) for name in SUPPORTED_VARIANTS}
    )
    assert not [name for name in names if RETIRED.search(name)]
    for name in ("dense_memory_attention", "strided_memory_attention", "dense_and_strided_memory_attention"):
        assert canonical_variant_name(name) == "memory_attention"
    assert variants.StridedSelfAttentionVariant.__module__.endswith(".strided_self_attention")
    assert variants.MemoryAttentionVariant.__module__.endswith(".memory_attention")
