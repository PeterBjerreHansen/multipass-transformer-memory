from __future__ import annotations

import torch

from tiny_mistral.config import MistralConfig


def micro_config(**overrides) -> MistralConfig:
    kwargs = dict(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        sliding_window=4,
        attention_dropout=0.0,
        torch_dtype="float32",
    )
    kwargs.update(overrides)
    return MistralConfig(**kwargs)


def pytest_configure(config):
    torch.manual_seed(1234)
