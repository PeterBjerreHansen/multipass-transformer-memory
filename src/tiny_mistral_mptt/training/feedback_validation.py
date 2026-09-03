"""Durable, idempotent feedback reports tied to a published snapshot."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from ..data.manifest import file_sha256
from ..evaluation.feedback import evaluate_feedback_nll, feedback_evaluation_metadata
from ..evaluation.settings import resolve_evaluation_settings
from ..evaluation.provenance import package_versions
from .durable import atomic_write_json
from .provenance import hardware_provenance


def evaluate_snapshot_feedback(
    model, dataset, *, snapshot: Path, config, device, source: dict, stop_requested,
) -> dict:
    """The caller guarantees live weights equal the just-published snapshot.

    Results are committed only after all selected full blocks finish. A changed
    request gets another report; it cannot silently reuse or overwrite old data.
    """
    settings = resolve_evaluation_settings(
        config, model, forward_mode="feedback",
        autocast_dtype=config.feedback_eval_autocast_dtype,
    )
    hardware = hardware_provenance(device)
    request = {
        "evaluator_version": 1,
        "snapshot_sha256": file_sha256(snapshot),
        "evaluation_source_code_sha256": source.get("source_code_sha256"),
        "package_versions": package_versions(),
        "runtime": {key: hardware.get(key) for key in ("torch", "cuda_runtime", "gpu_name")},
        "evaluation": feedback_evaluation_metadata(
            model, dataset, device=device, max_blocks=config.feedback_eval_max_blocks,
            autocast_dtype=settings.autocast_dtype,
        ),
    }
    report_id = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = snapshot.parent.parent / "evaluations" / f"feedback_{snapshot.stem}_{report_id[:16]}.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        result = report.get("result", {})
        if (report.get("request") != request or report.get("report_id") != report_id
                or result.get("evaluation") != request["evaluation"]
                or not all(isinstance(result.get(key), (int, float)) and math.isfinite(result[key])
                           for key in ("nll", "aligned_nll", "predicted_tokens", "aligned_predicted_tokens"))):
            raise ValueError(f"invalid committed feedback report: {path}")
    else:
        start = time.perf_counter()
        result = evaluate_feedback_nll(
            model, dataset, device=device, max_blocks=config.feedback_eval_max_blocks,
            autocast_dtype=settings.autocast_dtype, stop_requested=stop_requested,
        )
        report = {
            "schema_version": 2, "evaluation_kind": "bos_only_feedback_nll",
            "source": source, "hardware": hardware,
            "report_id": report_id, "request": request, "snapshot": str(snapshot.resolve()),
            "resolved_settings": asdict(settings), "result": asdict(result),
            "evaluation_elapsed_seconds": time.perf_counter() - start,
        }
        atomic_write_json(path, report)
    return {"report_path": str(path), **report}
