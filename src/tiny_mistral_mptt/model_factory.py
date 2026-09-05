from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from tiny_mistral.loading import load_model
from tiny_mistral.modeling import MistralForCausalLM

from .compatibility import normalize_legacy_variant_name
from .config import canonical_memory_write_mode, canonical_variant_name, resolve_memory_attention_pattern
from .variants import (
    ExperimentalVariant,
    FBTVariant,
    MemoryAddVariant,
    NoMemoryAdapterVariant,
    RecurrentMemoryVariant,
    MemoryAttentionRecurrentHybridVariant,
    MemoryAttentionVariant,
    StridedSelfAttentionVariant,
    SWATransformerVariant,
)

if TYPE_CHECKING:
    from .config import ExperimentConfig


def build_variant(
    name: str,
    backbone: MistralForCausalLM,
    *,
    architecture_seed: int = 4242,
    memory_window: int = 32,
    memory_pattern: str | None = None,
    memory_write_mode: str | None = None,
    memory_write_stride: int | None = None,
    memory_token_visibility: str | None = None,
    memory_layers: str | list[int] = "all",
    memory_position_encoding: str = "rope",
    memory_num_key_value_heads: int | None = None,
    memory_dense_window: int = 32,
    memory_sparse_window: int = 32,
    memory_sparse_stride: int = 32,
    sparse_attention_stride: int | None = None,
    sparse_attention_window: int | None = None,
    sparse_attention_layers: str | list[int] = "all",
    prefix_mixin_probability: float = 0.0,
    fbt_normalize_gate_input: bool = False,
    fbt_latent_jitter_std: float = 0.0,
    recirculation_source_layer: int | None = None,
    recirculation_destination_layer: int | None = None,
    recirculation_alpha: float = 0.1,
    recirculation_mode: str = "fixed",
    recurrent_merger: str | None = None,
    recurrent_controller_hidden_size: int | None = None,
    recurrent_layers: list[int] | None = None,
) -> ExperimentalVariant:
    requested_name = normalize_legacy_variant_name(str(name))
    name = canonical_variant_name(requested_name)
    if name == "recirculation":
        raise ValueError(
            "the middle-layer RecirculationVariant is archived; use "
            "variant=recurrent_memory with recurrent_merger=recirculation"
        )
    memory_write_mode = canonical_memory_write_mode(memory_write_mode)
    if name != "memory_attention":
        if memory_pattern is not None or recurrent_layers is not None:
            raise ValueError("memory_pattern and recurrent_layers apply only to Memory Attention")
        if recurrent_merger is not None and name != "recurrent_memory":
            raise ValueError("recurrent_merger requires recurrent_memory or Memory Attention")
    if name != "recirculation" and (
        recirculation_source_layer is not None or recirculation_destination_layer is not None
        or recirculation_alpha != 0.1 or recirculation_mode != "fixed"
    ):
        raise ValueError("recirculation_* fields apply only to legacy recirculation")
    if name == "vanilla":
        variant: ExperimentalVariant = SWATransformerVariant(backbone)
    elif name == "strided_self_attention":
        if sparse_attention_stride is None or sparse_attention_window is None:
            raise ValueError(
                "Strided Self-Attention requires sparse_attention_stride and "
                "sparse_attention_window"
            )
        variant = StridedSelfAttentionVariant(
            backbone,
            sparse_attention_stride=sparse_attention_stride,
            sparse_attention_window=sparse_attention_window,
            sparse_attention_layers=sparse_attention_layers,
        )
    elif name == "fbt":
        variant = FBTVariant(
            backbone,
            initialization_seed=architecture_seed,
            prefix_mixin_probability=prefix_mixin_probability,
            normalize_gate_input=fbt_normalize_gate_input,
            latent_jitter_std=fbt_latent_jitter_std,
        )
    elif name == "memory_add":
        variant = MemoryAddVariant(backbone)
    elif name == "no_memory_adapter":
        variant = NoMemoryAdapterVariant(
            backbone,
            memory_layers=memory_layers,
            initialization_seed=architecture_seed,
        )
    elif name == "recurrent_memory":
        variant = RecurrentMemoryVariant(
            backbone,
            memory_layers=memory_layers,
            merger=recurrent_merger,
            controller_hidden_size=recurrent_controller_hidden_size,
            initialization_seed=architecture_seed,
        )
    elif name == "memory_attention":
        memory_pattern, memory_write_mode = resolve_memory_attention_pattern(
            requested_name, memory_pattern, memory_write_mode
        )
        if memory_pattern == "dense_and_strided":
            if memory_write_stride is not None or memory_token_visibility is not None:
                raise ValueError("dense-and-strided retention does not accept memory_write_* controls")
            stride, visibility = 1, "visible"
        elif memory_write_mode == "dense":
            if memory_write_stride is not None or memory_token_visibility is not None:
                raise ValueError("dense Memory Attention must not set memory_write_stride or memory_token_visibility")
            stride, visibility = 1, "visible"
        else:
            if memory_write_stride is None or int(memory_write_stride) <= 0:
                raise ValueError("strided or memory-token Memory Attention requires positive memory_write_stride")
            stride = int(memory_write_stride)
            if memory_write_mode == "memory_token":
                if memory_token_visibility not in {"visible", "write_only"}:
                    raise ValueError("memory-token Memory Attention requires memory_token_visibility")
                visibility = str(memory_token_visibility)
            else:
                if memory_token_visibility is not None:
                    raise ValueError("memory_token_visibility applies only to memory_token mode")
                visibility = "visible"
        kwargs = dict(
            memory_pattern=memory_pattern,
            memory_window=memory_window,
            memory_write_mode="dense" if memory_pattern == "dense_and_strided" else memory_write_mode,
            memory_write_stride=stride,
            memory_token_visibility=visibility,
            memory_layers=memory_layers,
            memory_position_encoding=memory_position_encoding,
            memory_num_key_value_heads=memory_num_key_value_heads,
            memory_dense_window=memory_dense_window,
            memory_sparse_window=memory_sparse_window,
            memory_sparse_stride=memory_sparse_stride,
            initialization_seed=architecture_seed,
        )
        if recurrent_merger is None:
            if recurrent_layers is not None:
                raise ValueError("recurrent_layers requires recurrent_merger")
            variant = MemoryAttentionVariant(backbone, **kwargs)
        else:
            variant = MemoryAttentionRecurrentHybridVariant(
                backbone,
                recurrent_merger=recurrent_merger,
                recurrent_layers=recurrent_layers,
                recurrent_controller_hidden_size=recurrent_controller_hidden_size,
                **kwargs,
            )
    else:
        raise ValueError(f"unknown variant {name!r}")

    reference_parameter = next(backbone.parameters())
    variant.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
    # Preserve current public names, but never emit retired input aliases.
    variant.variant_name = requested_name
    return variant


def load_variant(
    name: str,
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype | None = None,
    attention_backend: str = "auto",
    compile_flex: bool = True,
    architecture_seed: int = 4242,
    memory_window: int = 32,
    memory_pattern: str | None = None,
    memory_write_mode: str | None = None,
    memory_write_stride: int | None = None,
    memory_token_visibility: str | None = None,
    memory_layers: str | list[int] = "all",
    memory_position_encoding: str = "rope",
    memory_num_key_value_heads: int | None = None,
    memory_dense_window: int = 32,
    memory_sparse_window: int = 32,
    memory_sparse_stride: int = 32,
    sparse_attention_stride: int | None = None,
    sparse_attention_window: int | None = None,
    sparse_attention_layers: str | list[int] = "all",
    prefix_mixin_probability: float = 0.0,
    fbt_normalize_gate_input: bool = False,
    fbt_latent_jitter_std: float = 0.0,
    recirculation_source_layer: int | None = None,
    recirculation_destination_layer: int | None = None,
    recirculation_alpha: float = 0.1,
    recirculation_mode: str = "fixed",
    recurrent_merger: str | None = None,
    recurrent_controller_hidden_size: int | None = None,
    recurrent_layers: list[int] | None = None,
) -> ExperimentalVariant:
    backbone = load_model(
        model_dir,
        device=device,
        dtype=dtype,
        attention_backend=attention_backend,
        compile_flex=compile_flex,
    )
    return build_variant(
        name,
        backbone,
        architecture_seed=architecture_seed,
        memory_window=memory_window,
        memory_pattern=memory_pattern,
        memory_write_mode=memory_write_mode,
        memory_write_stride=memory_write_stride,
        memory_token_visibility=memory_token_visibility,
        memory_layers=memory_layers,
        memory_position_encoding=memory_position_encoding,
        memory_num_key_value_heads=memory_num_key_value_heads,
        memory_dense_window=memory_dense_window,
        memory_sparse_window=memory_sparse_window,
        memory_sparse_stride=memory_sparse_stride,
        sparse_attention_stride=sparse_attention_stride,
        sparse_attention_window=sparse_attention_window,
        sparse_attention_layers=sparse_attention_layers,
        prefix_mixin_probability=prefix_mixin_probability,
        fbt_normalize_gate_input=fbt_normalize_gate_input,
        fbt_latent_jitter_std=fbt_latent_jitter_std,
        recirculation_source_layer=recirculation_source_layer,
        recirculation_destination_layer=recirculation_destination_layer,
        recirculation_alpha=recirculation_alpha,
        recirculation_mode=recirculation_mode,
        recurrent_merger=recurrent_merger,
        recurrent_controller_hidden_size=recurrent_controller_hidden_size,
        recurrent_layers=recurrent_layers,
    )


def _architecture_kwargs(cfg: "ExperimentConfig") -> dict:
    """Translate one normalized experiment config at the factory boundary."""
    return dict(
        architecture_seed=cfg.architecture_seed,
        memory_window=cfg.memory_window,
        memory_pattern=cfg.memory_pattern,
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
        memory_token_visibility=cfg.memory_token_visibility,
        memory_layers="all" if cfg.memory_layers is None else cfg.memory_layers,
        memory_position_encoding=(
            "rope" if cfg.memory_position_encoding is None else cfg.memory_position_encoding
        ),
        memory_num_key_value_heads=cfg.memory_num_key_value_heads,
        memory_dense_window=(
            32 if cfg.memory_dense_window is None else cfg.memory_dense_window
        ),
        memory_sparse_window=(
            32 if cfg.memory_sparse_window is None else cfg.memory_sparse_window
        ),
        memory_sparse_stride=(
            32 if cfg.memory_sparse_stride is None else cfg.memory_sparse_stride
        ),
        sparse_attention_stride=cfg.sparse_attention_stride,
        sparse_attention_window=cfg.sparse_attention_window,
        sparse_attention_layers=(
            "all"
            if cfg.sparse_attention_layers is None
            else cfg.sparse_attention_layers
        ),
        prefix_mixin_probability=cfg.prefix_mixin_probability,
        fbt_normalize_gate_input=cfg.fbt_normalize_gate_input,
        fbt_latent_jitter_std=cfg.fbt_latent_jitter_std,
        recirculation_source_layer=cfg.recirculation_source_layer,
        recirculation_destination_layer=cfg.recirculation_destination_layer,
        recirculation_alpha=cfg.recirculation_alpha,
        recirculation_mode=cfg.recirculation_mode,
        recurrent_merger=cfg.recurrent_merger,
        recurrent_controller_hidden_size=cfg.recurrent_controller_hidden_size,
        recurrent_layers=cfg.recurrent_layers,
    )


def build_variant_from_config(
    cfg: "ExperimentConfig",
    backbone: MistralForCausalLM,
) -> ExperimentalVariant:
    """Build configured wiring around an already-created backbone."""
    return build_variant(cfg.variant, backbone, **_architecture_kwargs(cfg))


def load_variant_from_config(
    cfg: "ExperimentConfig",
    *,
    device: str | torch.device | None = None,
) -> ExperimentalVariant:
    backbone = load_model(
        cfg.model_dir,
        device=cfg.device if device is None else device,
        dtype=cfg.dtype,
        attention_backend=cfg.attention_backend,
    )
    return build_variant_from_config(cfg, backbone)
