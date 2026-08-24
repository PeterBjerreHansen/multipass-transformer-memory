"""Public recirculation + Memory Attention compatibility import."""

from .bank_recirculation_hybrid import BankRecirculationHybridVariant

MemoryAttentionRecirculationHybridVariant = BankRecirculationHybridVariant

__all__ = [
    "BankRecirculationHybridVariant",
    "MemoryAttentionRecirculationHybridVariant",
]
