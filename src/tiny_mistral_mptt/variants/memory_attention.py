"""Public Memory Attention names for the historical bank implementation."""

from .bank import BankBatch, BankReader, BankVariant, BankWriter

MemoryAttentionBatch = BankBatch
MemoryAttentionReader = BankReader
MemoryAttentionVariant = BankVariant
MemoryAttentionWriter = BankWriter

__all__ = [
    "BankBatch",
    "BankReader",
    "BankVariant",
    "BankWriter",
    "MemoryAttentionBatch",
    "MemoryAttentionReader",
    "MemoryAttentionVariant",
    "MemoryAttentionWriter",
]
