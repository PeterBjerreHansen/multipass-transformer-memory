from .base import ExperimentalVariant, TrainOutput
from .fbt import FBTVariant
from .memory_add import MemoryAddVariant
from .multipass import HiddenRun, MultiPassResult, MultiPassVariant, PassResult, shift_previous_hidden
from .recirculation import RecirculationVariant
from .recurrent_memory import RecurrentMemoryVariant
from .bank import (
    BankBatch,
    BankReader,
    BankVariant,
    BankWriter,
    MemoryAttentionBatch,
    MemoryAttentionReader,
    MemoryAttentionVariant,
    MemoryAttentionWriter,
)
from .bank_add_hybrid import BankAddHybridVariant, MemoryAttentionAddHybridVariant
from .bank_recirculation_hybrid import (
    BankRecirculationHybridVariant,
    RecirculationStridedMemoryAttentionVariant,
)
from .bank_recurrent_hybrid import (
    BankRecurrentHybridVariant,
    MemoryAttentionRecurrentHybridVariant,
)
from .bank_multiscale import MultiscaleBankVariant, MultiscaleMemoryAttentionVariant
from .sparse_swa import SparseSWAVariant, StridedAttentionVariant
from .vanilla import SWATransformerVariant, SwaTransformerVariant, VanillaVariant

MemoryAttentionRecirculationHybridVariant = RecirculationStridedMemoryAttentionVariant

__all__ = [
    "ExperimentalVariant",
    "FBTVariant",
    "MemoryAddVariant",
    "HiddenRun",
    "MultiPassResult",
    "MultiPassVariant",
    "PassResult",
    "RecirculationVariant",
    "RecurrentMemoryVariant",
    "BankAddHybridVariant",
    "BankBatch",
    "BankReader",
    "BankRecirculationHybridVariant",
    "BankRecurrentHybridVariant",
    "BankVariant",
    "MultiscaleBankVariant",
    "BankWriter",
    "MemoryAttentionBatch",
    "MemoryAttentionRecurrentHybridVariant",
    "RecirculationStridedMemoryAttentionVariant",
    "SwaTransformerVariant",
    "SWATransformerVariant",
    "StridedAttentionVariant",
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
