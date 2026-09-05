import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.inference import exact_decode_step, prefill_exact_k_pass
from tiny_mistral_mptt.model_factory import build_variant
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.no_memory_adapter import NoMemoryAdapterVariant


def make_model(layers=(0,)):
    torch.manual_seed(811)
    return NoMemoryAdapterVariant(
        MistralForCausalLM(
            micro_config(num_hidden_layers=2), attention_backend="reference"
        ),
        memory_layers=list(layers),
        initialization_seed=92,
    )


def test_adapter_is_feedback_pass_only_and_ignores_previous_pass_state():
    model = make_model().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    embeddings = model.input_embeddings(ids)
    previous_a = torch.randn_like(embeddings)
    previous_b = torch.randn_like(embeddings)
    with torch.no_grad():
        first = model.compute_passes(ids, passes=1).final.logits
        backbone = model.backbone(ids, use_cache=False).logits
        feedback_a = model._run_feedback_hidden(ids, embeddings, previous_a)
        feedback_b = model._run_feedback_hidden(ids, embeddings, previous_b)
    torch.testing.assert_close(first, backbone, atol=0, rtol=0)
    torch.testing.assert_close(feedback_a, feedback_b, atol=0, rtol=0)


def test_adapter_parameters_enter_the_native_gradient_schedule():
    model = make_model()
    configure_phase(model, "A")
    optimizer = torch.optim.SGD(model.added_parameters(), lr=0.1)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    seen = set()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model.compute_loss(
            ids, phase="A", passes=2, loss_weights=[0.0, 1.0]
        ).loss.backward()
        seen.update(
            id(parameter)
            for parameter in model.added_parameters()
            if parameter.grad is not None and bool(parameter.grad.ne(0).any())
        )
        optimizer.step()
    assert seen == {id(parameter) for parameter in model.added_parameters()}


@pytest.mark.parametrize("layers", ([0], [0, 1]))
def test_matched_arms_are_within_ten_percent_on_instantiated_parameters(layers):
    def backbone():
        return MistralForCausalLM(
            micro_config(num_hidden_layers=2), attention_backend="reference"
        )

    models = {
        "no_memory": build_variant(
            "no_memory_adapter", backbone(), memory_layers=layers
        ),
        "projected": build_variant(
            "recurrent_memory",
            backbone(),
            memory_window=1,
            memory_layers=layers,
            recurrent_merger="projected_residual",
        ),
        "recirculation_inspired": build_variant(
            "recurrent_memory",
            backbone(),
            memory_window=1,
            memory_layers=layers,
            recurrent_merger="recirculation",
            recurrent_controller_hidden_size=20,
        ),
        "dense_attention": build_variant(
            "dense_memory_attention",
            backbone(),
            memory_window=4,
            memory_layers=layers,
            memory_num_key_value_heads=2,
        ),
    }
    counts = {
        name: sum(parameter.numel() for parameter in model.added_parameters())
        for name, model in models.items()
    }
    assert max(counts.values()) / min(counts.values()) < 1.1, counts


def test_ambiguous_public_execution_fails():
    model = make_model()
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(RuntimeError, match="no unambiguous temporal semantics"):
        model(ids)
    with pytest.raises(RuntimeError, match="no unambiguous temporal semantics"):
        model.generate(ids, 1)


def test_cached_exact_k_pass_matches_full_recomputation():
    model = make_model().eval()
    with torch.no_grad():
        model.memory_mergers["0"].projection.weight.copy_(
            0.03 * torch.eye(model.config.hidden_size)
        )
    ids = torch.tensor([[1, 7, 3, 14, 22, 9, 31, 4]])
    prompt_length = 3
    with torch.no_grad():
        state = prefill_exact_k_pass(
            model, ids[:, :prompt_length], passes=3
        )
        for position in range(prompt_length, ids.shape[1]):
            state = exact_decode_step(model, state, ids[:, position : position + 1])
            full = model.compute_passes(ids[:, : position + 1], passes=3)
            torch.testing.assert_close(
                state.next_token_logits,
                full.final.logits[:, -1, :],
                atol=4e-5,
                rtol=4e-5,
            )
