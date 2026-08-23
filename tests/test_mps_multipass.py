import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.fbt import FBTVariant
from tiny_mistral_mptt.variants.memory_add import MemoryAddVariant
from tiny_mistral_mptt.variants.recirculation import RecirculationVariant
from tiny_mistral_mptt.variants.bank import BankVariant


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS hardware/runtime unavailable",
)


@pytest.mark.parametrize("variant_name", ["fbt", "memory_add", "bank"])
def test_multipass_variants_forward_backward_on_mps(variant_name):
    config = micro_config()
    backbone = MistralForCausalLM(config, attention_backend="auto").to("mps", dtype=torch.float32)
    if variant_name == "fbt":
        model = FBTVariant(backbone, initialization_seed=17)
    elif variant_name == "memory_add":
        model = MemoryAddVariant(backbone)
    else:
        model = BankVariant(backbone, memory_window=4, memory_write_mode="dense", memory_write_stride=1, initialization_seed=17)
    model = model.to("mps", dtype=torch.float32)
    configure_phase(model, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="mps")
    output = model.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    assert torch.isfinite(output.loss)
    output.loss.backward()
    grads = [parameter.grad for parameter in model.added_parameters() if parameter.grad is not None]
    assert grads
    assert all(bool(torch.isfinite(grad).all().item()) for grad in grads)


@pytest.mark.parametrize("variant_name", ["memory_add", "bank", "recirculation"])
@pytest.mark.parametrize("passes", [2, 3])
def test_incremental_memory_inference_on_mps(variant_name, passes):
    from tiny_mistral_mptt.inference import (
        exact_decode_step,
        prefill_exact,
        recurrent_decode_step,
        recurrent_from_exact,
    )

    config = micro_config(sliding_window=4)
    backbone = MistralForCausalLM(
        config, attention_backend="auto"
    ).to("mps", dtype=torch.float32)
    if variant_name == "memory_add":
        model = MemoryAddVariant(backbone)
        with torch.no_grad():
            model.memory_projection.weight.copy_(
                0.05 * torch.eye(config.hidden_size, device="mps")
            )
    elif variant_name == "bank":
        model = BankVariant(
            backbone, memory_window=3, memory_write_mode="dense", memory_write_stride=1, initialization_seed=17
        )
    else:
        model = RecirculationVariant(
            backbone, source_layer=1, destination_layer=0, alpha=0.1
        )
    model = model.to("mps", dtype=torch.float32).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="mps")

    exact = prefill_exact(model, ids[:, :5], passes=passes)
    recurrent = recurrent_from_exact(exact)
    for position in (5, 6):
        token = ids[:, position : position + 1]
        exact = exact_decode_step(model, exact, token)
        recurrent = recurrent_decode_step(model, recurrent, token)
        assert bool(torch.isfinite(exact.next_token_logits).all().item())
        assert bool(torch.isfinite(recurrent.next_token_logits).all().item())
        assert exact.next_position == recurrent.next_position == position + 1
