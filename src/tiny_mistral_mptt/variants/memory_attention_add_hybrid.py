"""Public Memory Attention + residual hybrid compatibility import."""

from .bank_add_hybrid import BankAddHybridVariant

MemoryAttentionAddHybridVariant = BankAddHybridVariant

__all__ = ["BankAddHybridVariant", "MemoryAttentionAddHybridVariant"]
