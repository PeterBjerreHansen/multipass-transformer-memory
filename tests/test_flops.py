from tiny_mistral.config import tiny_mistral_248m_config
from tiny_mistral_mptt.flops import (
    _tape_pairs,
    estimate_pass,
    estimate_schedule,
    memory_token_layout,
    tape_write_positions,
)


def test_memory_token_layout_matches_packed_dataset_contract():
    layout = memory_token_layout(2048, 32)
    assert len(layout) == 2111
    assert sum(layout) == 63
    uses_controls, writes, key_layout = tape_write_positions(
        linguistic_length=2048,
        memory_write_mode="memory_token",
        memory_write_stride=32,
    )
    assert uses_controls is True
    assert len(writes) == 63
    assert key_layout == layout


def test_tape_pairs_are_strictly_past_and_windowed():
    # Query positions 0..3 see 0, 0, 1, and 1 strictly prior writes. The write
    # at position 3 is not visible to the query at position 3.
    assert _tape_pairs(4, (1, 3), 2) == 2


def test_stage5_flop_estimates_include_architecture_specific_work():
    config = tiny_mistral_248m_config()
    baseline = estimate_schedule(
        config,
        variant="vanilla",
        pass_probabilities={1: 1.0},
        linguistic_sequence_length=2048,
    )
    dense = estimate_schedule(
        config,
        variant="tape",
        pass_probabilities={2: 0.9, 3: 0.1},
        linguistic_sequence_length=2048,
        memory_window=32,
        memory_write_mode="dense",
        memory_layers=[3, 7],
    )
    periodic = estimate_schedule(
        config,
        variant="tape",
        pass_probabilities={2: 0.9, 3: 0.1},
        linguistic_sequence_length=2048,
        memory_window=32,
        memory_write_mode="periodic",
        memory_write_stride=32,
        memory_layers=[3, 7],
    )
    memory_token = estimate_schedule(
        config,
        variant="tape",
        pass_probabilities={2: 0.9, 3: 0.1},
        linguistic_sequence_length=2048,
        memory_window=32,
        memory_write_mode="memory_token",
        memory_write_stride=32,
        memory_token_visibility="write_only",
        memory_layers=[3, 7],
    )

    assert baseline.relative_training_flops == 1.0
    assert dense.relative_training_flops > 2.0
    assert periodic.relative_training_flops < dense.relative_training_flops
    assert memory_token.relative_training_flops > periodic.relative_training_flops
    assert memory_token.pass_estimates[2].physical_sequence_length == 2111
    assert memory_token.pass_estimates[2].memory_positions == 63


def test_fixed_recursion_does_not_charge_adaptive_controller():
    config = tiny_mistral_248m_config()
    fixed = estimate_pass(
        config,
        variant="recirculation",
        passes=2,
        linguistic_sequence_length=2048,
        recirculation_mode="fixed",
    )
    adaptive = estimate_pass(
        config,
        variant="recirculation",
        passes=2,
        linguistic_sequence_length=2048,
        recirculation_mode="adaptive",
    )
    assert fixed.forward.recurrent_controller == 0
    assert adaptive.forward.recurrent_controller > 0
