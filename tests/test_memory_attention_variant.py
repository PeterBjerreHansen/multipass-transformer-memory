import pytest
import torch

from conftest import micro_config
from tiny_mistral.modeling import MistralForCausalLM
from tiny_mistral_mptt.feedback import HybridFeedbackState, MemoryAttentionState
from tiny_mistral_mptt.training.loss import causal_lm_loss_from_labels
from tiny_mistral_mptt.training.phases import configure_phase
from tiny_mistral_mptt.variants.memory_attention import MemoryAttentionVariant
from tiny_mistral_mptt.variants.memory_attention_recurrent_hybrid import MemoryAttentionRecurrentHybridVariant


def backbone(seed=123, *, backend="reference", sliding_window=8):
    torch.manual_seed(seed)
    return MistralForCausalLM(
        micro_config(num_hidden_layers=2, sliding_window=sliding_window),
        attention_backend=backend,
        compile_flex=False,
    )


def memory_model(
    *,
    mode="periodic",
    stride=2,
    window=4,
    hybrid=False,
    seed=123,
    backend="reference",
):
    cls = MemoryAttentionRecurrentHybridVariant if hybrid else MemoryAttentionVariant
    model = cls(
        backbone(seed, backend=backend),
        **({"recurrent_merger": "projected_residual", "recurrent_layers": [0]} if hybrid else {}),
        memory_window=window,
        memory_write_mode=mode,
        memory_write_stride=stride,
        initialization_seed=991,
    )
    if hybrid:
        with torch.no_grad():
            model.memory_mergers["0"].projection.weight.copy_(0.03 * torch.eye(model.config.hidden_size))
    return model


def activate_memory_readers(model):
    with torch.no_grad():
        for reader in model.memory_readers.values():
            reader.o_proj.weight.copy_(torch.eye(model.config.hidden_size))


def test_dense_and_periodic_c1_are_the_same_memory_architecture():
    dense = memory_model(mode="dense", stride=1, seed=5).eval()
    periodic = memory_model(mode="periodic", stride=1, seed=99).eval()
    periodic.backbone.load_state_dict(dense.backbone.state_dict())
    periodic.writer.load_state_dict(dense.writer.state_dict())
    periodic.memory_readers.load_state_dict(dense.memory_readers.state_dict())
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        for passes in (2, 3):
            expected = dense.compute_passes(ids, passes=passes).final.hidden_states
            actual = periodic.compute_passes(ids, passes=passes).final.hidden_states
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_zero_initialized_memory_is_exact_vanilla_at_every_pass_depth():
    model = memory_model(mode="periodic", stride=2).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        outputs = model.compute_passes(ids, passes=3)
    for later in outputs.passes[1:]:
        torch.testing.assert_close(
            later.logits, outputs.passes[0].logits, atol=0, rtol=0
        )


def test_aligned_gqa_initialization_is_nonzero_and_group_aligned():
    model = MemoryAttentionVariant(
        backbone(seed=5),
        memory_window=4,
        memory_write_mode="dense",
        memory_layers=[0],
        memory_position_encoding="none",
        memory_num_key_value_heads=2,
        memory_reader_initialization="aligned_gqa",
        initialization_seed=991,
    ).eval()
    reader = model.memory_readers["0"]
    group_size = reader.num_heads // reader.num_key_value_heads

    assert torch.count_nonzero(reader.o_proj.weight) > 0
    for kv_head in range(reader.num_key_value_heads):
        first_query_head = kv_head * group_size
        for group_offset in range(1, group_size):
            other_query_head = first_query_head + group_offset
            first = reader.q_proj.weight[
                first_query_head * reader.head_dim : (first_query_head + 1) * reader.head_dim
            ]
            other = reader.q_proj.weight[
                other_query_head * reader.head_dim : (other_query_head + 1) * reader.head_dim
            ]
            torch.testing.assert_close(first, other, atol=0, rtol=0)
        key = reader.k_proj.weight[
            kv_head * reader.head_dim : (kv_head + 1) * reader.head_dim
        ]
        query = reader.q_proj.weight[
            first_query_head * reader.head_dim : (first_query_head + 1) * reader.head_dim
        ]
        torch.testing.assert_close(query, key, atol=0, rtol=0)

    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        passes = model.compute_passes(ids, passes=2)
    torch.testing.assert_close(
        passes.passes[0].hidden_states[:, 0],
        passes.passes[1].hidden_states[:, 0],
        atol=0,
        rtol=0,
    )
    assert not torch.allclose(
        passes.passes[0].hidden_states[:, 1:],
        passes.passes[1].hidden_states[:, 1:],
    )


@pytest.mark.parametrize(
    "fusion", ["destination_gated", "attention_gated", "dual_gated"]
)
def test_attention_fusion_controllers_are_added_parameters(fusion):
    model = MemoryAttentionVariant(
        backbone(seed=5),
        memory_window=4,
        memory_write_mode="dense",
        memory_layers=[0],
        memory_position_encoding="none",
        memory_reader_initialization="aligned_gqa",
        memory_attention_fusion=fusion,
        memory_attention_controller_hidden_size=4,
        initialization_seed=991,
    )
    destination = torch.randn(2, 3, model.config.hidden_size)
    memory = torch.randn_like(destination)
    adapter = model.memory_attention_fusions["0"]
    gates = adapter.gate_values(memory, destination)
    if fusion in {"attention_gated", "dual_gated"}:
        assert gates.alpha is not None
        torch.testing.assert_close(gates.alpha, torch.full_like(gates.alpha, 0.1))
    else:
        assert gates.alpha is None
    if fusion in {"destination_gated", "dual_gated"}:
        assert gates.beta is not None
        torch.testing.assert_close(gates.beta, torch.full_like(gates.beta, 0.9))
    else:
        assert gates.beta is None

    available = torch.tensor([[False, True, True], [False, False, True]])
    fused = model._fuse_memory(0, destination, memory, available=available)
    torch.testing.assert_close(fused[~available], destination[~available], atol=0, rtol=0)
    assert not torch.allclose(fused[available], destination[available])
    added = {id(parameter) for parameter in model.added_parameters()}
    assert all(id(parameter) in added for parameter in adapter.parameters())


def test_periodic_write_mask_uses_completed_stride_positions():
    model = memory_model(mode="periodic", stride=4)
    ids = torch.arange(10)[None, :]
    assert model.write_mask(ids).tolist() == [[
        False, False, False, True, False, False, False, True, False, False
    ]]


def test_dense_write_mask_writes_every_position():
    model = memory_model(mode="dense", stride=1)
    ids = torch.arange(6)[None, :]
    assert model.write_mask(ids).all()


def test_periodic_write_is_strict_past():
    model = memory_model(mode="periodic", stride=4).eval()
    activate_memory_readers(model)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    embeddings = model.input_embeddings(ids)
    previous = torch.randn_like(embeddings)
    perturbed = previous.clone()
    perturbed[:, 3, :] += 3.0
    with torch.no_grad():
        base = model._run_feedback_hidden(ids, embeddings, previous)
        changed = model._run_feedback_hidden(ids, embeddings, perturbed)
    # The record written at position 3 cannot affect query position 3 itself.
    torch.testing.assert_close(changed[:, :4, :], base[:, :4, :], atol=0, rtol=0)
    assert not torch.allclose(changed[:, 4:, :], base[:, 4:, :])


def test_phase_a_noop_initialization_stages_reader_then_writer_gradients():
    model = memory_model(mode="periodic")
    configure_phase(model, "A")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    optimizer = torch.optim.SGD(list(model.added_parameters()), lr=0.1)
    output = model.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()
    for reader in model.memory_readers.values():
        assert reader.o_proj.weight.grad is not None
        assert reader.o_proj.weight.grad.abs().sum() > 0
    assert model.writer.proj.weight.grad is not None
    assert model.writer.proj.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    output = model.compute_loss(ids, phase="A", passes=2, loss_weights=[0.0, 1.0])
    output.loss.backward()
    assert model.writer.proj.weight.grad is not None
    assert model.writer.proj.weight.grad.abs().sum() > 0
    added = {id(parameter) for parameter in model.added_parameters()}
    for parameter in model.parameters():
        if id(parameter) not in added:
            assert parameter.grad is None


def test_seeded_periodic_memory_keeps_last_w_records():
    model = memory_model(mode="periodic", stride=2, window=3).eval()
    hidden = torch.arange(1 * 8 * model.config.hidden_size, dtype=torch.float32).reshape(
        1, 8, model.config.hidden_size
    )
    ids = torch.arange(8)[None, :]
    state = model._feedback_memory_from_hidden(hidden, input_ids=ids)
    assert isinstance(state, MemoryAttentionState)
    assert state.valid.tolist() == [[True, True, True]]
    torch.testing.assert_close(state.memories[0], hidden[0, [3, 5, 7], :])
