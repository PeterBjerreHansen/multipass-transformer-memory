"""Strided self-attention control exports."""

from .sparse_swa import SparseSWAVariant, StridedAttentionVariant

__all__ = ["StridedAttentionVariant", "SparseSWAVariant"]
