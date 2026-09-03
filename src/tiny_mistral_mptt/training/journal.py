from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .checkpoint import TrainState
from .durable import fsync_directory


def append_jsonl(path: str | Path, item: dict[str, Any], *, durable: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")
        if durable:
            handle.flush()
            os.fsync(handle.fileno())


def _record_is_committed(record: dict[str, Any], state: TrainState) -> bool:
    event = record.get("event")
    if event == "train":
        try:
            return int(record["optimizer_steps"]) <= state.optimizer_steps
        except (KeyError, TypeError, ValueError):
            return False
    if event in {"validation", "feedback_validation"}:
        try:
            return int(record["unique_tokens_seen"]) <= state.unique_tokens_seen
        except (KeyError, TypeError, ValueError):
            return False
    if event in {"resume", "snapshot"}:
        try:
            return int(record.get("unique_tokens_seen", 0)) <= state.unique_tokens_seen
        except (TypeError, ValueError):
            return False
    return True


def event_recorded(path: Path, event: str, **identity) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (isinstance(record, dict) and record.get("event") == event
                    and all(record.get(key) == value for key, value in identity.items())):
                return True
    return False


def repair_metrics_to_checkpoint(path: str | Path, state: TrainState) -> dict[str, int]:
    """Roll an append-only metrics journal back to the durable checkpoint."""
    path = Path(path)
    if not path.exists():
        return {"kept": 0, "dropped": 0, "malformed": 0}

    kept: list[str] = []
    dropped = 0
    malformed = 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(record, dict) and _record_is_committed(record, state):
                kept.append(json.dumps(record, sort_keys=True))
            else:
                dropped += 1

    temporary = path.with_name(path.name + ".repair.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for line in kept:
            handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)
    return {"kept": len(kept), "dropped": dropped, "malformed": malformed}
