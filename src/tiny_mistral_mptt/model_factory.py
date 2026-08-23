from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from tiny_mistral.loading import load_model
from tiny_mistral.modeling import MistralForCausalLM

from .variants import (
    ExperimentalVariant,
    FBTVariant,
    MemoryAddVariant,
    RecirculationVariant,
    BankAddHybridVariant,
    BankRecirculationHybridVariant,
    MultiscaleBankVariant,
    BankVariant,
    SparseSWAVariant,
    VanillaVariant,
)

if TYPE_CHECKING:
    from .config import ExperimentConfig


def build_variant(
    name: str,
    backbone: MistralForCausalLM,
    *,
    architecture_seed: int = 4242,
    memory_window: int = 32,
    memory_write_mode: str | None = None,
    memory_write_stride: int | None = None,
    memory_token_visibility: str | None = None,
    memory_layers: str | list[int] = "all",
    memory_position_encoding: str = "rope",
    memory_dense_window: int = 32,
    memory_sparse_window: int = 32,
    memory_sparse_stride: int = 32,
    sparse_attention_stride: int | None = None,
    sparse_attention_window: int | None = None,
    sparse_attention_layers: str | list[int] = "all",
    prefix_mixin_probability: float = 0.0,
    recirculation_source_layer: int | None = None,
    recirculation_destination_layer: int | None = None,
    recirculation_alpha: float = 0.1,
    recirculation_mode: str = "fixed",
    recurrent_nmp_weight: float = 0.0,
    bank_nmp_weight: float = 0.0,
    recurrent_nmp_target_normalization: str = "rms",
    nmp_projection_factor: float = 1.3,
) -> ExperimentalVariant:
    if name == "vanilla":
        variant: ExperimentalVariant = VanillaVariant(backbone)
    elif name == "sparse_swa":
        if sparse_attention_stride is None or sparse_attention_window is None:
            raise ValueError(
                "sparse_swa requires sparse_attention_stride and "
                "sparse_attention_window"
            )
        variant = SparseSWAVariant(
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
        )
    elif name == "memory_add":
        variant = MemoryAddVariant(backbone)
    elif name == "recirculation":
        if recirculation_source_layer is None or recirculation_destination_layer is None:
            raise ValueError(
                "recirculation requires recirculation_source_layer and "
                "recirculation_destination_layer"
            )
        variant = RecirculationVariant(
            backbone,
            source_layer=recirculation_source_layer,
            destination_layer=recirculation_destination_layer,
            alpha=recirculation_alpha,
            mode=recirculation_mode,
            initialization_seed=architecture_seed,
        )
    elif name == "bank_multiscale":
        if (
            memory_write_mode is not None
            or memory_write_stride is not None
            or memory_token_visibility is not None
        ):
            raise ValueError(
                "bank_multiscale uses dense source states and does not accept "
                "memory_write_* controls"
            )
        variant = MultiscaleBankVariant(
            backbone,
            memory_dense_window=memory_dense_window,
            memory_sparse_window=memory_sparse_window,
            memory_sparse_stride=memory_sparse_stride,
            memory_layers=memory_layers,
            memory_position_encoding=memory_position_encoding,
            initialization_seed=architecture_seed,
        )
    elif name in {"bank", "bank_add_hybrid", "bank_recirculation_hybrid"}:
        if memory_write_mode not in {"dense", "periodic", "memory_token"}:
            raise ValueError("bank variants require memory_write_mode: dense|periodic|memory_token")
        if memory_write_mode == "dense":
            if memory_write_stride is not None:
                raise ValueError("dense bank must not set memory_write_stride")
            if memory_token_visibility is not None:
                raise ValueError("dense bank must not set memory_token_visibility")
            stride = 1
            visibility = "visible"
        elif memory_write_mode == "periodic":
            if memory_write_stride is None or int(memory_write_stride) <= 0:
                raise ValueError("periodic bank requires positive memory_write_stride")
            if memory_token_visibility is not None:
                raise ValueError("memory_token_visibility applies only to memory_token mode")
            stride = int(memory_write_stride)
            visibility = "visible"
        else:
            if memory_write_stride is None or int(memory_write_stride) <= 0:
                raise ValueError("memory_token bank requires positive memory_write_stride")
            if memory_token_visibility not in {"visible", "write_only"}:
                raise ValueError("memory_token bank requires memory_token_visibility: visible|write_only")
            stride = int(memory_write_stride)
            visibility = str(memory_token_visibility)
        kwargs = dict(
            memory_window=memory_window,
            memory_write_mode=memory_write_mode,
            memory_write_stride=stride,
            memory_token_visibility=visibility,
            memory_layers=memory_layers,
            memory_position_encoding=memory_position_encoding,
            initialization_seed=architecture_seed,
        )
        if name == "bank":
            variant = BankVariant(backbone, **kwargs)
        elif name == "bank_add_hybrid":
            variant = BankAddHybridVariant(backbone, **kwargs)
        else:
            if (
                recirculation_source_layer is None
                or recirculation_destination_layer is None
            ):
                raise ValueError(
                    "bank_recirculation_hybrid requires recirculation_source_layer "
                    "and recirculation_destination_layer"
                )
            variant = BankRecirculationHybridVariant(
                backbone,
                source_layer=recirculation_source_layer,
                destination_layer=recirculation_destination_layer,
                alpha=recirculation_alpha,
                mode=recirculation_mode,
                **kwargs,
            )
    else:
        raise ValueError(f"unknown variant {name!r}")

    if recurrent_nmp_weight or bank_nmp_weight:
        from .variants import MultiPassVariant

        if not isinstance(variant, MultiPassVariant):
            raise ValueError(f"{name} does not support NMP")
        variant.configure_nmp(
            recurrent_weight=recurrent_nmp_weight,
            bank_weight=bank_nmp_weight,
            recurrent_target_normalization=recurrent_nmp_target_normalization,
            projection_factor=nmp_projection_factor,
            initialization_seed=architecture_seed,
        )

    reference_parameter = next(backbone.parameters())
    variant.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
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
    memory_write_mode: str | None = None,
    memory_write_stride: int | None = None,
    memory_token_visibility: str | None = None,
    memory_layers: str | list[int] = "all",
    memory_position_encoding: str = "rope",
    memory_dense_window: int = 32,
    memory_sparse_window: int = 32,
    memory_sparse_stride: int = 32,
    sparse_attention_stride: int | None = None,
    sparse_attention_window: int | None = None,
    sparse_attention_layers: str | list[int] = "all",
    prefix_mixin_probability: float = 0.0,
    recirculation_source_layer: int | None = None,
    recirculation_destination_layer: int | None = None,
    recirculation_alpha: float = 0.1,
    recirculation_mode: str = "fixed",
    recurrent_nmp_weight: float = 0.0,
    bank_nmp_weight: float = 0.0,
    recurrent_nmp_target_normalization: str = "rms",
    nmp_projection_factor: float = 1.3,
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
        memory_write_mode=memory_write_mode,
        memory_write_stride=memory_write_stride,
        memory_token_visibility=memory_token_visibility,
        memory_layers=memory_layers,
        memory_position_encoding=memory_position_encoding,
        memory_dense_window=memory_dense_window,
        memory_sparse_window=memory_sparse_window,
        memory_sparse_stride=memory_sparse_stride,
        sparse_attention_stride=sparse_attention_stride,
        sparse_attention_window=sparse_attention_window,
        sparse_attention_layers=sparse_attention_layers,
        prefix_mixin_probability=prefix_mixin_probability,
        recirculation_source_layer=recirculation_source_layer,
        recirculation_destination_layer=recirculation_destination_layer,
        recirculation_alpha=recirculation_alpha,
        recirculation_mode=recirculation_mode,
        recurrent_nmp_weight=recurrent_nmp_weight,
        bank_nmp_weight=bank_nmp_weight,
        recurrent_nmp_target_normalization=recurrent_nmp_target_normalization,
        nmp_projection_factor=nmp_projection_factor,
    )


def load_variant_from_config(
    cfg: "ExperimentConfig",
    *,
    device: str | torch.device | None = None,
) -> ExperimentalVariant:
    return load_variant(
        cfg.variant,
        cfg.model_dir,
        device=cfg.device if device is None else device,
        dtype=cfg.dtype,
        attention_backend=cfg.attention_backend,
        architecture_seed=cfg.architecture_seed,
        memory_window=cfg.memory_window,
        memory_write_mode=cfg.memory_write_mode,
        memory_write_stride=cfg.memory_write_stride,
        memory_token_visibility=cfg.memory_token_visibility,
        memory_layers="all" if cfg.memory_layers is None else cfg.memory_layers,
        memory_position_encoding=(
            "rope" if cfg.memory_position_encoding is None else cfg.memory_position_encoding
        ),
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
        recirculation_source_layer=cfg.recirculation_source_layer,
        recirculation_destination_layer=cfg.recirculation_destination_layer,
        recirculation_alpha=cfg.recirculation_alpha,
        recirculation_mode=cfg.recirculation_mode,
        recurrent_nmp_weight=cfg.recurrent_nmp_weight,
        bank_nmp_weight=cfg.bank_nmp_weight,
        recurrent_nmp_target_normalization=cfg.recurrent_nmp_target_normalization,
        nmp_projection_factor=cfg.nmp_projection_factor,
    )
