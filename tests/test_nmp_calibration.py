import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/development/nmp_100m_continuation/calibrate_nmp.py"
)
_SPEC = importlib.util.spec_from_file_location("nmp_calibration", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_weighted_gradients = _MODULE._weighted_gradients
_weighted_metric = _MODULE._weighted_metric


def test_mixed_pass_calibration_combines_exact_configured_probabilities():
    outputs = {
        2: SimpleNamespace(
            metrics={"ntp_loss": 4.0},
        ),
        3: SimpleNamespace(
            metrics={"ntp_loss": 36.0},
        ),
    }
    probabilities = {2: 0.9, 3: 0.1}
    gradients = {
        2: {"pretrained": [torch.tensor([4.0, 2.0])]},
        3: {"pretrained": [torch.tensor([36.0, -2.0])]},
    }

    mixture = _weighted_gradients(gradients, probabilities)

    torch.testing.assert_close(
        mixture["pretrained"][0], torch.tensor([7.2, 1.6])
    )
    assert _weighted_metric(outputs, probabilities, "ntp_loss") == 7.2
