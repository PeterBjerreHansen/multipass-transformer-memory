import torch
from conftest import micro_config

from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.fbt import FBTVariant
from tiny_mistral_mptt.variants.multipass import shift_previous_hidden


def make_variant():
    torch.manual_seed(10)
    backbone = MistralForCausalLM(micro_config(), attention_backend="reference")
    return FBTVariant(backbone, initialization_seed=123)


def test_fbt_one_pass_is_exact_vanilla():
    variant = make_variant().eval()
    ids = torch.tensor([[1, 4, 8, 12, 16]])
    with torch.no_grad():
        direct = variant.backbone(ids, use_cache=False).logits
        pass_one = variant.compute_passes(ids, passes=1).passes[0].logits
    torch.testing.assert_close(pass_one, direct, atol=0, rtol=0)


def test_fbt_feedback_uses_strict_previous_position_and_keeps_position_zero():
    variant = make_variant()
    dim = variant.config.hidden_size
    with torch.no_grad():
        variant.feedback_value.weight.copy_(torch.eye(dim))
        variant.feedback_gate.weight.zero_()  # sigmoid(0) == 0.5
    embeddings = torch.randn(1, 4, dim)
    previous = torch.randn(1, 4, dim)
    fused = variant.feedback_inputs(embeddings, previous)
    torch.testing.assert_close(fused[:, 0], embeddings[:, 0])
    torch.testing.assert_close(
        fused[:, 1:], variant.feedback_input_norm(0.5 * previous[:, :-1])
    )


def test_fbt_phase_a_trains_only_added_parameters_and_has_gradient():
    variant = make_variant()
    configure_phase(variant, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = variant.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()

    added_ids = {id(parameter) for parameter in variant.added_parameters()}
    assert any(parameter.grad is not None for parameter in variant.added_parameters())
    for parameter in variant.parameters():
        if id(parameter) not in added_ids:
            assert parameter.grad is None


def test_fbt_phase_b_later_pass_loss_backpropagates_through_previous_pass():
    variant = make_variant()
    configure_phase(variant, "B")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = variant.compute_loss(ids, phase="B", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()
    assert variant.backbone.model.embed_tokens.weight.grad is not None
    assert torch.isfinite(variant.backbone.model.embed_tokens.weight.grad).all()


def test_fbt_prefix_mixin_reverts_a_checkpoint_reproducible_prefix():
    variant = make_variant()
    variant.prefix_mixin_probability = 1.0
    dim = variant.config.hidden_size
    embeddings = torch.randn(1, 5, dim)
    previous = torch.randn(1, 5, dim)
    with torch.no_grad():
        variant.feedback_value.weight.copy_(torch.eye(dim))
        variant.feedback_gate.weight.zero_()
        shifted = shift_previous_hidden(previous)
        raw = torch.cat(
            (
                embeddings[:, :1, :],
                variant.feedback_input_norm(0.5 * shifted[:, 1:, :]),
            ),
            dim=1,
        )

    torch.manual_seed(99)
    expected_prefix = int(
        torch.randint(1, embeddings.shape[1] + 1, (), device="cpu").item()
    )
    torch.manual_seed(99)
    mixed = variant.feedback_inputs(embeddings, previous)
    torch.testing.assert_close(
        mixed[:, :expected_prefix, :], embeddings[:, :expected_prefix, :]
    )
    torch.testing.assert_close(
        mixed[:, expected_prefix:, :], raw[:, expected_prefix:, :]
    )


def test_fbt_paper_gate_normalizes_embeddings_before_projection():
    variant = make_variant().eval()
    variant.normalize_gate_input = True
    dim = variant.config.hidden_size
    embeddings = torch.randn(1, 4, dim)
    previous = torch.randn(1, 4, dim)
    with torch.no_grad():
        variant.feedback_value.weight.copy_(torch.eye(dim))
        variant.feedback_gate.weight.copy_(torch.eye(dim))
    shifted = shift_previous_hidden(previous)
    expected = variant.feedback_input_norm(
        shifted
        * torch.sigmoid(variant.feedback_input_norm(embeddings))
    )

    fused = variant.feedback_inputs(embeddings, previous)

    torch.testing.assert_close(fused[:, 0], embeddings[:, 0])
    torch.testing.assert_close(fused[:, 1:], expected[:, 1:])


def test_fbt_latent_jitter_is_uniform_and_training_only():
    variant = make_variant()
    variant.latent_jitter_std = 0.02
    dim = variant.config.hidden_size
    embeddings = torch.randn(1, 4, dim)
    previous = torch.randn(1, 4, dim)
    with torch.no_grad():
        variant.feedback_value.weight.copy_(torch.eye(dim))
        variant.feedback_gate.weight.zero_()

    torch.manual_seed(2026)
    noise = torch.empty_like(previous).uniform_(-0.02, 0.02)
    expected = variant.feedback_input_norm(
        0.5 * shift_previous_hidden(previous + noise)
    )
    torch.manual_seed(2026)
    jittered = variant.feedback_inputs(embeddings, previous)
    torch.testing.assert_close(jittered[:, 1:], expected[:, 1:])

    variant.eval()
    unjittered = variant.feedback_inputs(embeddings, previous)
    expected_eval = variant.feedback_input_norm(
        0.5 * shift_previous_hidden(previous)
    )
    torch.testing.assert_close(unjittered[:, 1:], expected_eval[:, 1:])
    assert not torch.equal(jittered[:, 1:], unjittered[:, 1:])
