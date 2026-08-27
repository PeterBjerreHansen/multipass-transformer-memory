#!/usr/bin/env python
"""Hardware smoke test for the Apple-MPS attention backends.

Uses a tiny random Mistral configuration, so no checkpoint download is needed.
"""
from __future__ import annotations

import argparse

import torch

from tiny_mistral.config import MistralConfig
from tiny_mistral.device import mps_available, synchronize
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants import MultiscaleMemoryAttentionVariant


def config() -> MistralConfig:
    return MistralConfig(
        vocab_size=257,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        sliding_window=8,
        attention_dropout=0.0,
        torch_dtype="float16",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not mps_available():
        raise SystemExit(
            "MPS is unavailable. Check that this is an Apple Silicon Mac, that your macOS/PyTorch "
            "build supports MPS, and that torch.backends.mps.is_available() returns True."
        )

    device = torch.device("mps")
    cfg = config()
    ref = MistralForCausalLM(cfg, attention_backend="reference").to(device=device, dtype=torch.float16).eval()
    local = MistralForCausalLM(cfg, attention_backend="local").to(device=device, dtype=torch.float16).eval()
    local.load_state_dict(ref.state_dict())
    ids = torch.randint(0, cfg.vocab_size, (1, 33), device=device)

    with torch.no_grad():
        a = ref(ids, use_cache=False).logits
        b = local(ids, use_cache=False).logits
    synchronize(device)
    diff = (a - b).abs()
    print(f"forward max_abs_diff={diff.max().item():.8g} mean_abs_diff={diff.mean().item():.8g}")
    torch.testing.assert_close(b, a, atol=3e-2, rtol=3e-2)

    ref.set_sparse_attention(stride=8, window=4, layers=[0])
    local.set_sparse_attention(stride=8, window=4, layers=[0])
    with torch.no_grad():
        sparse_reference = ref(ids, use_cache=False).logits
        sparse_local = local(ids, use_cache=False).logits
    synchronize(device)
    sparse_diff = (sparse_reference - sparse_local).abs()
    print(
        "sparse forward "
        f"max_abs_diff={sparse_diff.max().item():.8g} "
        f"mean_abs_diff={sparse_diff.mean().item():.8g}"
    )
    torch.testing.assert_close(
        sparse_local, sparse_reference, atol=3e-2, rtol=3e-2
    )

    model = local.train()
    model.zero_grad(set_to_none=True)
    out = model(ids, labels=ids, use_cache=False)
    assert out.loss is not None and bool(torch.isfinite(out.loss).item())
    out.loss.backward()
    if not any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("no gradients produced")
    synchronize(device)
    print(f"loss={out.loss.item():.6f}")

    memory_backbone = MistralForCausalLM(
        cfg, attention_backend="local"
    ).to(device=device, dtype=torch.float16)
    memory_backbone.load_state_dict(ref.state_dict())
    memory_attention = MultiscaleMemoryAttentionVariant(
        memory_backbone,
        memory_dense_window=8,
        memory_sparse_window=4,
        memory_sparse_stride=8,
        memory_layers=[0],
    ).to(device=device, dtype=torch.float16).train()
    configure_phase(memory_attention, "A")
    memory_output = memory_attention.compute_loss(
        ids,
        phase="A",
        passes=2,
        loss_weights=[0.0, 1.0],
    )
    if not bool(torch.isfinite(memory_output.loss).item()):
        raise RuntimeError("non-finite Multiscale Memory Attention loss")
    memory_output.loss.backward()
    if not any(
        parameter.grad is not None
        and bool(parameter.grad.detach().ne(0).any().item())
        for parameter in memory_attention.added_parameters()
        if parameter.requires_grad
    ):
        raise RuntimeError("Multiscale Memory Attention produced no added-parameter gradients")
    synchronize(device)
    print(f"multiscale_memory_attention_loss={memory_output.loss.item():.6f}")
    print("PASS: MPS local, Strided Attention, and Multiscale Memory Attention smoke tests")


if __name__ == "__main__":
    main()
