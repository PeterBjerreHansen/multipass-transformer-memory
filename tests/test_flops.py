import json
from pathlib import Path
import subprocess
import sys

from tiny_mistral.config import tiny_mistral_248m_config
from tiny_mistral_mptt.flops import (
    _memory_pairs,
    estimate_pass,
    estimate_schedule,
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
                / "large"
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
    assert {
        "adapter_baseline_100m",
        "dense_memory_attention_residual_100m",
        "recurrent_recirculation_100m",
        "recurrent_projected_residual_100m",
        "strided_memory_attention_stride8_100m",
        "dense_and_strided_memory_attention_stride8_100m",
        "dense_memory_attention_destination_gated_aligned_100m",
        "dense_memory_attention_residual_aligned_100m",
        "dense_memory_attention_attention_gated_aligned_100m",
        "dense_memory_attention_dual_gated_aligned_100m",
    } == set(rows)
    assert {
        (row["batch_size"], row["grad_accum_steps"])
        for row in rows.values()
        if "dense_and_strided" not in row["arm"]
    } == {(8, 4)}
    combined = rows["dense_and_strided_memory_attention_stride8_100m"]
    assert (combined["batch_size"], combined["grad_accum_steps"]) == (4, 8)
    assert {row["optimizer_batch_tokens"] for row in rows.values()} == {65_536}
    assert all(row["estimated_training_flops_total"] > 0 for row in rows.values())
    assert rows["recurrent_recirculation_100m"]["training_forward"] == "parallel_multipass"
    assert rows["recurrent_recirculation_100m"]["relative_training_flops"] > 1.0


def test_wiring_budget_report_instantiates_matched_groups_and_stride_spans():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_wiring_budgets.py"),
            "--study",
            str(
                ROOT
                / "benchmarks"
                / "development"
                / "frozen_backbone_comparison"
                / "large"
                / "STUDY.yaml"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert all(
        group["within_ten_percent"]
        for group in report["matched_groups"].values()
    )
    rows = {row["arm"]: row for row in report["arms"]}
    assert rows["strided_memory_attention_stride8_100m"]["physical_write_count"] == 256
    assert rows["strided_memory_attention_stride8_100m"]["effective_memory_span_tokens"] == 256
    assert rows["dense_and_strided_memory_attention_stride8_100m"]["physical_write_count"] == 2048


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
    assert baseline.relative_training_flops == 1.0
    assert dense.relative_training_flops > 2.0
    assert periodic.relative_training_flops < dense.relative_training_flops


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


def test_attention_fusion_controller_cost_tracks_one_or_two_gate_heads():
    config = tiny_mistral_248m_config()
    common = dict(
        variant="dense_memory_attention",
        passes=2,
        linguistic_sequence_length=128,
        memory_layers=[3, 7],
        memory_num_key_value_heads=16,
    )
    residual = estimate_pass(config, **common)
    destination = estimate_pass(
        config,
        memory_attention_fusion="destination_gated",
        memory_attention_controller_hidden_size=64,
        **common,
    )
    attention = estimate_pass(
        config,
        memory_attention_fusion="attention_gated",
        memory_attention_controller_hidden_size=64,
        **common,
    )
    dual = estimate_pass(
        config,
        memory_attention_fusion="dual_gated",
        memory_attention_controller_hidden_size=64,
        **common,
    )
    assert destination.forward.memory_writer == residual.forward.memory_writer
    assert destination.forward.memory_reader_projections == residual.forward.memory_reader_projections
    assert destination.forward.memory_reader_products == residual.forward.memory_reader_products
    assert residual.forward.memory_fusion_controller == 0
    assert destination.forward.memory_fusion_controller > 0
    assert attention.forward.memory_fusion_controller == destination.forward.memory_fusion_controller
    assert dual.forward.memory_fusion_controller > destination.forward.memory_fusion_controller
    assert dual.forward.recurrent_controller == 0
