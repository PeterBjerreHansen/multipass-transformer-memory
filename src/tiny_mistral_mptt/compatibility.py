"""Input-only translations for serialized experiment names.

These names are not implementation keys or public exports. Keep recorded
artifacts unchanged; translate their identifiers when loading/comparing them.
"""

LEGACY_VARIANT_ALIASES = {
    "bank": "memory_attention",
    "bank_multiscale": "dense_and_strided_memory_attention",
    "multiscale_memory_attention": "dense_and_strided_memory_attention",
    "memory_attention_multiscale": "dense_and_strided_memory_attention",
    "attention_dense_and_strided": "dense_and_strided_memory_attention",
    "strided_attention": "strided_self_attention",
    "sparse_swa": "strided_self_attention",
}

_CHECKPOINT_ONLY_ALIASES = {
    "tape": "memory_attention",
    "tape_multiscale": "dense_and_strided_memory_attention",
}

REMOVED_HYBRID_VARIANTS = {
    "memory_attention_add_hybrid", "memory_attention_recirculation_hybrid",
    "recirculation_strided_memory_attention", "bank_add_hybrid",
    "bank_recirculation_hybrid", "tape_add_hybrid", "tape_recirculation_hybrid",
}


def normalize_legacy_variant_name(name: str) -> str:
    """Translate a retired config name before normal validation or dispatch."""
    if name in REMOVED_HYBRID_VARIANTS:
        raise ValueError(
            f"{name!r} has been removed; the optional late-memory hybrid is a "
            "different architecture and cannot reuse its checkpoints"
        )
    return LEGACY_VARIANT_ALIASES.get(name, name)


def normalize_checkpoint_variant_name(name: str) -> str:
    """Also accept identifiers restricted to old checkpoint metadata."""
    return normalize_legacy_variant_name(_CHECKPOINT_ONLY_ALIASES.get(name, name))
