import json
from pathlib import Path
import subprocess
import sys

from tiny_mistral.config import tiny_mistral_248m_config
from tiny_mistral_mptt.flops import (
    _memory_pairs,
    estimate_pass,
    estimate_schedule,
    memory_token_layout,
    memory_write_positions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_study_flop_report_uses_authoritative_arm_batching():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "estimate_training_flops.py"),
            "--study",
            str(
                ROOT
                / "benchmarks"
                / "development"
                / "frozen_backbone_comparison"
                / "STUDY.yaml"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    rows = {row["arm"]: row for row in report["results"]}
    assert set(rows) == {
        "recurrent_recirculation_multipass_100m",
        "recurrent_projected_residual_multipass_100m",
        "dense_memory_attention_multipass_100m",
        "strided_memory_attention_multipass_100m",
        "dense_and_strided_memory_attention_multipass_100m",
    }
    assert {
        (row["batch_size"], row["grad_accum_steps"])
        for row in rows.values()
    } == {(8, 4)}
    assert {row["optimizer_batch_tokens"] for row in rows.values()} == {65_536}
    assert all(row["estimated_training_flops_total"] > 0 for row in rows.values())
    assert rows["recurrent_recirculation_multipass_100m"]["training_forward"] == "parallel_multipass"
    assert rows["recurrent_recirculation_multipass_100m"]["relative_training_flops"] > 1.0


def test_recurrent_memory_counts_shared_writer_and_each_merger():
    config = tiny_mistral_248m_config()
    linear = 2 * 128 * config.hidden_size ** 2
    for merger, controller_factor, projection_factor in (
        ("recirculation", 5, 0), ("projected_residual", 2, 1)
    ):
        estimate = estimate_pass(
            config, variant="recurrent_memory", passes=3,
            linguistic_sequence_length=128, memory_layers=[3, 7],
            recurrent_merger=merger,
        )
        assert estimate.forward.memory_writer == 2 * linear
        assert estimate.forward.recurrent_controller == 2 * 2 * controller_factor * linear
        assert estimate.forward.recurrent_projection == 2 * 2 * projection_factor * linear


def test_memory_token_layout_matches_packed_dataset_contract():
    layout = memory_token_layout(2048, 32)
    assert len(layout) == 2111
    assert sum(layout) == 63
    uses_controls, writes, key_layout = memory_write_positions(
        linguistic_length=2048,
        memory_write_mode="memory_token",
        memory_write_stride=32,
    )
    assert uses_controls is True
    assert len(writes) == 63
    assert key_layout == layout


def test_memory_pairs_are_strictly_past_and_windowed():
    # Query positions 0..3 see 0, 0, 1, and 1 strictly prior writes. The write
    # at position 3 is not visible to the query at position 3.
    assert _memory_pairs(4, (1, 3), 2) == 2


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
        variant="memory_attention",
        pass_probabilities={2: 0.9, 3: 0.1},
        linguistic_sequence_length=2048,
        memory_window=32,
        memory_write_mode="dense",
        memory_layers=[3, 7],
    )
    periodic = estimate_schedule(
        config,
        variant="memory_attention",
        pass_probabilities={2: 0.9, 3: 0.1},
        linguistic_sequence_length=2048,
        memory_window=32,
        memory_write_mode="periodic",
        memory_write_stride=32,
        memory_layers=[3, 7],
    )
    memory_token = estimate_schedule(
        config,
        variant="memory_attention",
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


def test_strided_self_attention_adds_only_sparse_attention_products():
    config = tiny_mistral_248m_config()
    vanilla = estimate_pass(
        config,
        variant="vanilla",
        passes=1,
        linguistic_sequence_length=128,
    )
    sparse = estimate_pass(
        config,
        variant="strided_self_attention",
        passes=1,
        linguistic_sequence_length=128,
        sparse_attention_stride=32,
        sparse_attention_window=2,
        sparse_attention_layers=[3, 7],
    )
    assert sparse.forward.self_attention_projections == vanilla.forward.self_attention_projections
    assert sparse.forward.self_attention_products > vanilla.forward.self_attention_products
    assert sparse.forward.mlp_projections == vanilla.forward.mlp_projections


def test_dense_and_strided_memory_flops_count_dense_writes_and_union_reads():
    config = tiny_mistral_248m_config()
    estimate = estimate_pass(
        config,
        variant="dense_and_strided_memory_attention",
        passes=2,
        linguistic_sequence_length=128,
        memory_dense_window=32,
        memory_sparse_window=2,
        memory_sparse_stride=32,
        memory_layers=[4, 7],
    )
    assert estimate.memory_write_positions == 128
    assert estimate.memory_positions == 0
    assert estimate.forward.memory_writer > 0
    assert estimate.forward.memory_reader_products > 0


def test_optional_hybrid_counts_both_writer_applications_and_shared_mergers():
    config = tiny_mistral_248m_config()
    for merger in ("projected_residual", "recirculation"):
        common = dict(passes=3, linguistic_sequence_length=128)
        attention = estimate_pass(config, variant="dense_memory_attention", memory_layers=[3, 7], **common)
        recurrent = estimate_pass(config, variant="recurrent_memory", memory_layers=[3],
                                  recurrent_merger=merger, **common)
        hybrid = estimate_pass(config, variant="dense_memory_attention", memory_layers=[3, 7],
                               recurrent_merger=merger, recurrent_layers=[3], **common)
        assert hybrid.forward.memory_writer == attention.forward.memory_writer + recurrent.forward.memory_writer
        assert hybrid.forward.memory_reader_projections == attention.forward.memory_reader_projections
        assert hybrid.forward.recurrent_controller == recurrent.forward.recurrent_controller
        assert hybrid.forward.recurrent_projection == recurrent.forward.recurrent_projection
