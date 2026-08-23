import pytest

from tiny_mistral_mptt.efficiency import recommend_cuda_microbatch


def _row(variant: str, batch: int, rate: float, *, status: str = "ok") -> dict:
    row = {
        "variant": variant,
        "passes": 2,
        "sequence_length": 2048,
        "batch_size": batch,
        "grad_accum_steps": 1,
        "autocast_dtype": "bfloat16",
        "device": "cuda:0",
        "status": status,
    }
    if status == "ok":
        row["unique_tokens_per_second"] = rate
    return row


def test_recommend_cuda_microbatch_uses_smallest_common_efficient_batch():
    document = {
        "results": [
            _row("recirculation", 1, 100.0),
            _row("recirculation", 2, 180.0),
            _row("recirculation", 4, 200.0),
            _row("recirculation", 8, 205.0),
            _row("tape", 1, 80.0),
            _row("tape", 2, 150.0),
            _row("tape", 4, 155.0),
            _row("tape", 8, 0.0, status="oom"),
        ]
    }
    result = recommend_cuda_microbatch(document, efficiency_fraction=0.90)
    assert result.selected_microbatch == 4
    assert result.common_successful_microbatches == (1, 2, 4)
    assert result.optimizer_batch_tokens == 8192
    assert result.reference_optimizer_batch_tokens == 2048
    assert result.changes_optimizer_batch is True
    assert result.local_grad_accum_steps_to_match is None
    assert result.throughput_fraction_by_variant["recirculation"] == pytest.approx(200 / 205)
    assert result.throughput_fraction_by_variant["tape"] == pytest.approx(1.0)


def test_recommendation_reports_accumulation_when_reference_batch_is_larger():
    document = {
        "results": [
            _row("recirculation", 1, 100.0),
            _row("recirculation", 2, 100.0),
            _row("tape", 1, 100.0),
            _row("tape", 2, 100.0),
        ]
    }
    result = recommend_cuda_microbatch(
        document,
        efficiency_fraction=0.90,
        reference_optimizer_batch_tokens=8192,
    )
    assert result.selected_microbatch == 1
    assert result.optimizer_batch_tokens == 2048
    assert result.local_grad_accum_steps_to_match == 4


def test_recommend_cuda_microbatch_rejects_missing_variant_rows():
    document = {"results": [_row("recirculation", 1, 100.0)]}
    with pytest.raises(ValueError, match="tape"):
        recommend_cuda_microbatch(document)


def test_recommend_cuda_microbatch_ignores_accumulated_rows():
    row = _row("tape", 1, 100.0)
    row["grad_accum_steps"] = 2
    document = {"results": [_row("recirculation", 1, 100.0), row]}
    with pytest.raises(ValueError, match="tape"):
        recommend_cuda_microbatch(document)
