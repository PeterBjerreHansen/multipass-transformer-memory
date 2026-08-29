import torch

from tiny_mistral_mptt.evaluation.drift import parameter_drift_summary


def test_parameter_drift_separates_backbone_and_added_parameters():
    reference = {
        "backbone.weight": torch.tensor([1.0, 2.0]),
        "memory.weight": torch.tensor([3.0]),
    }
    current = {
        "backbone.weight": torch.tensor([2.0, 2.0]),
        "memory.weight": torch.tensor([5.0]),
    }
    result = parameter_drift_summary(
        reference,
        current,
        added_names={"memory.weight"},
    )
    assert result["groups"]["backbone"]["parameters"] == 2
    assert result["groups"]["added"]["parameters"] == 1
    assert result["groups"]["all"]["rms_delta"] == (5.0 / 3.0) ** 0.5
