from .base import ExperimentalVariant, TrainOutput
from .fbt import FBTVariant
from .memory_add import MemoryAddVariant
from .multipass import HiddenRun, MultiPassResult, MultiPassVariant, PassResult, shift_previous_hidden
from .no_memory_adapter import NoMemoryAdapterVariant
from .recurrent_memory import RecurrentMemoryVariant
from .memory_attention import (
    MemoryAttentionBatch,
    MemoryAttentionReader,
    MemoryAttentionVariant,
    MemoryAttentionWriter,
)
from .memory_attention_recurrent_hybrid import (
    MemoryAttentionRecurrentHybridVariant,
)
from .strided_self_attention import StridedSelfAttentionVariant
from .vanilla import SWATransformerVariant, SwaTransformerVariant, VanillaVariant


__all__ = [
    "ExperimentalVariant",
    "FBTVariant",
    "MemoryAddVariant",
    "HiddenRun",
    "MultiPassResult",
    "MultiPassVariant",
    "PassResult",
    "NoMemoryAdapterVariant",
    "RecurrentMemoryVariant",
    "MemoryAttentionBatch",
    "MemoryAttentionReader",
    "MemoryAttentionRecurrentHybridVariant",
    "MemoryAttentionVariant",
    "MemoryAttentionWriter",
    "SwaTransformerVariant",
    "SWATransformerVariant",
    "StridedSelfAttentionVariant",
    "TrainOutput",
    "VanillaVariant",
    "shift_previous_hidden",
]
