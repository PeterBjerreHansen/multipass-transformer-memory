from .base import ExperimentalVariant, TrainOutput
from .fbt import FBTVariant
from .memory_add import MemoryAddVariant
from .multipass import HiddenRun, MultiPassResult, MultiPassVariant, PassResult, shift_previous_hidden
from .recirculation import RecirculationVariant
from .bank import BankBatch, BankReader, BankVariant, BankWriter
from .bank_add_hybrid import BankAddHybridVariant
from .bank_recirculation_hybrid import BankRecirculationHybridVariant
from .bank_recurrent_hybrid import BankRecurrentHybridVariant
from .bank_multiscale import MultiscaleBankVariant
from .sparse_swa import SparseSWAVariant
from .vanilla import VanillaVariant

# Public Memory Attention vocabulary.  The Bank* classes remain the concrete
# implementation names so historical imports and checkpoint provenance stay
# stable.
MemoryAttentionVariant = BankVariant
MemoryAttentionReader = BankReader
MemoryAttentionWriter = BankWriter
MultiscaleMemoryAttentionVariant = MultiscaleBankVariant
MemoryAttentionAddHybridVariant = BankAddHybridVariant
MemoryAttentionRecirculationHybridVariant = BankRecirculationHybridVariant

__all__ = [
    "ExperimentalVariant",
    "FBTVariant",
    "MemoryAddVariant",
    "HiddenRun",
    "MultiPassResult",
    "MultiPassVariant",
    "PassResult",
    "RecirculationVariant",
    "BankAddHybridVariant",
    "BankBatch",
    "BankReader",
    "BankRecirculationHybridVariant",
    "BankRecurrentHybridVariant",
    "BankVariant",
    "MultiscaleBankVariant",
    "BankWriter",
    "TrainOutput",
    "VanillaVariant",
    "SparseSWAVariant",
    "shift_previous_hidden",
    "MemoryAttentionAddHybridVariant",
    "MemoryAttentionReader",
    "MemoryAttentionRecirculationHybridVariant",
    "MemoryAttentionVariant",
    "MemoryAttentionWriter",
    "MultiscaleMemoryAttentionVariant",
]
