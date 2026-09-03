from __future__ import annotations

from tiny_mistral.modeling import MistralForCausalLM

from .vanilla import SWATransformerVariant


class StridedSelfAttentionVariant(SWATransformerVariant):
    """Single-pass Transformer control with sparse fixed-position SWA keys.

    Selected decoder layers reuse their pretrained self-attention projections
    and apply one softmax over ordinary local SWA keys plus a bounded set of
    older strided keys. The variant adds no parameters and no cross-attention.
    """

    variant_name = "strided_self_attention"

    def __init__(
        self,
        backbone: MistralForCausalLM,
        *,
        sparse_attention_stride: int,
        sparse_attention_window: int,
        sparse_attention_layers: str | list[int] = "all",
    ):
        super().__init__(backbone)
        if int(sparse_attention_stride) <= 0:
            raise ValueError("sparse_attention_stride must be positive")
        if int(sparse_attention_window) <= 0:
            raise ValueError("sparse_attention_window must be positive")
        selected = backbone.set_sparse_attention(
            stride=int(sparse_attention_stride),
            window=int(sparse_attention_window),
            layers=sparse_attention_layers,
        )
        self.sparse_attention_stride = int(sparse_attention_stride)
        self.sparse_attention_window = int(sparse_attention_window)
        self.sparse_attention_layers = selected

__all__ = ["StridedSelfAttentionVariant"]
