from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time
from typing import Callable

import torch
from safetensors.torch import save_file as save_safetensors

from tiny_mistral.device import synchronize

from ..config import ExperimentConfig
from ..data.manifest import file_sha256, verify_artifact
from ..data.packed_dataset import MemoryTokenPackedDataset, PackedTokenDataset, StatefulBlockSampler
from ..evaluation.nll import evaluate_nll
from ..evaluation.pass_depth import evaluate_pass_depth
from ..precision import autocast_context
from ..variants.base import ExperimentalVariant
from ..variants.multipass import MultiPassVariant
from .checkpoint import (
    TrainState,
    inspect_checkpoint,
    load_checkpoint,
    load_latest_valid_checkpoint,
    load_model_weights,
    save_checkpoint_generation,
)
from .journal import append_jsonl, repair_metrics_to_checkpoint
from .pass_schedule import PassScheduler
from .phases import configure_phase
from .provenance import hardware_provenance, source_provenance
from .schedule import lr_multiplier


Dataset = PackedTokenDataset | MemoryTokenPackedDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _next_segment_id(path: Path) -> int:
    maximum = 0
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    maximum = max(maximum, int(record.get("segment", 0)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    return maximum + 1


def _validation_stop_status(record: dict, gates: dict) -> dict:
    """Evaluate declarative pass-depth gates against one validation record."""
    nll_by_pass = [float(value) for value in record.get("nll_by_pass", [])]
    hidden_deltas = [float(value) for value in record.get("hidden_delta_rms", [])]

    pass_nll_max: dict[str, dict[str, float | bool]] = {}
    for pass_index, threshold in gates["pass_nll_max"].items():
        actual = nll_by_pass[int(pass_index) - 1]
        pass_nll_max[str(pass_index)] = {
            "actual": actual,
            "max": float(threshold),
            "passed": actual <= float(threshold),
        }

    pass_nll_delta_max: list[dict[str, float | int | bool]] = []
    for gate in gates["pass_nll_delta_max"]:
        pass_index = int(gate["pass"])
        reference_pass = int(gate["reference_pass"])
        actual_delta = nll_by_pass[pass_index - 1] - nll_by_pass[reference_pass - 1]
        pass_nll_delta_max.append(
            {
                "pass": pass_index,
                "reference_pass": reference_pass,
                "actual_delta": actual_delta,
                "max_delta": float(gate["max_delta"]),
                "passed": actual_delta <= float(gate["max_delta"]),
            }
        )

    hidden_status = None
    if gates["hidden_delta_nonincreasing"]:
        hidden_status = {
            "actual": hidden_deltas,
            "passed": all(
                later <= earlier
                for earlier, later in zip(hidden_deltas, hidden_deltas[1:])
            ),
        }

    outcomes = [item["passed"] for item in pass_nll_max.values()]
    outcomes.extend(item["passed"] for item in pass_nll_delta_max)
    if hidden_status is not None:
        outcomes.append(hidden_status["passed"])
    return {
        "all_passed": bool(outcomes) and all(outcomes),
        "pass_nll_max": pass_nll_max,
        "pass_nll_delta_max": pass_nll_delta_max,
        "hidden_delta_nonincreasing": hidden_status,
    }


class Trainer:
    def __init__(
        self,
        *,
        model: ExperimentalVariant,
        config: ExperimentConfig,
        train_data: Dataset,
        validation_data: Dataset,
        device: torch.device,
        resume_auto: bool = False,
        allow_source_mismatch: bool = False,
        stop_requested: Callable[[], bool] | None = None,
    ):
        config.validate()
        if resume_auto and config.resume_from:
            raise ValueError("resume_auto is mutually exclusive with resume_from")
        if train_data.manifest != validation_data.manifest:
            raise ValueError("train and validation datasets must come from the same artifact")
        if train_data.sequence_length != validation_data.sequence_length:
            raise ValueError("train and validation physical sequence lengths differ")
        if train_data.sequence_length > int(model.config.max_position_embeddings):
            raise ValueError(
                "physical model sequence (including control slots) exceeds max_position_embeddings"
            )
        if train_data.manifest.vocab_size != int(model.config.vocab_size):
            raise ValueError("data base vocabulary differs from model output vocabulary")

        self.model = model
        self.config = config
        self.train_data = train_data
        self.validation_data = validation_data
        self.device = device
        self.resume_auto = bool(resume_auto)
        self.allow_source_mismatch = bool(allow_source_mismatch)
        self.stop_requested = stop_requested or (lambda: False)

        self.run_dir = Path(config.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.segments_path = self.run_dir / "segments.jsonl"
        self.run_info_path = self.run_dir / "run.json"
        run_info_preexisting = self.run_info_path.exists()

        self.repository = Path(__file__).resolve().parents[3]
        self.source = source_provenance(self.repository)
        self.hardware = hardware_provenance(device)

        verify_artifact(config.data_dir)
        self.manifest_path = Path(config.data_dir) / "manifest.json"
        self.manifest_sha256 = file_sha256(self.manifest_path)

        # Memory-token views expand physical positions but preserve a constant
        # number of linguistic/data tokens in every backing block. Avoid
        # per-microbatch GPU synchronizations by deriving this from the view.
        self.linguistic_per_block = int(
            getattr(train_data, "linguistic_sequence_length", train_data.sequence_length)
        )
        validation_linguistic = int(
            getattr(validation_data, "linguistic_sequence_length", validation_data.sequence_length)
        )
        if self.linguistic_per_block <= 0 or validation_linguistic != self.linguistic_per_block:
            raise ValueError("train/validation linguistic block lengths must match and be positive")
        # One eager semantic check catches a mismatched model/view before paid work.
        probe = train_data.batch([0], device="cpu")
        if model.linguistic_token_count(probe) != self.linguistic_per_block:
            raise ValueError("dataset control-token layout disagrees with model semantics")

        _set_seed(config.seed)
        self.sampler = StatefulBlockSampler(len(train_data), seed=config.seed + 1)
        self.pass_scheduler = PassScheduler(config.pass_schedule, seed=config.seed + 2)

        initialization_provenance = None
        if config.init_from:
            initialization_provenance = load_model_weights(
                config.init_from,
                model=self.model,
                expected_experiment_config=config.to_dict(),
            )
            initialization_provenance["source_sha256"] = file_sha256(config.init_from)

        trainable = configure_phase(model, config.phase)
        if trainable == 0:
            raise RuntimeError(f"Phase {config.phase} has no trainable parameters")
        self.optimizer = self._build_optimizer()
        self.state = TrainState(phase=config.phase)

        added_ids = {id(parameter) for parameter in model.added_parameters()}
        trainable_pretrained = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in added_ids
        ]
        trainable_added = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) in added_ids
        ]
        microbatch_tokens = config.batch_size * self.linguistic_per_block
        microbatch_positions = config.batch_size * train_data.sequence_length
        nominal_optimizer_batch_tokens = microbatch_tokens * config.grad_accum_steps
        nominal_optimizer_batch_positions = microbatch_positions * config.grad_accum_steps
        planned_optimizer_steps = (
            config.max_unique_tokens + nominal_optimizer_batch_tokens - 1
        ) // nominal_optimizer_batch_tokens
        run_info = {
            "source": self.source,
            "config": config.to_dict(),
            "batching": {
                "linguistic_sequence_length": self.linguistic_per_block,
                "physical_sequence_length": train_data.sequence_length,
                "microbatch_size": config.batch_size,
                "grad_accum_steps": config.grad_accum_steps,
                "microbatch_tokens": microbatch_tokens,
                "microbatch_model_positions": microbatch_positions,
                "control_positions_per_microbatch": microbatch_positions - microbatch_tokens,
                "nominal_optimizer_batch_tokens": nominal_optimizer_batch_tokens,
                "nominal_optimizer_batch_model_positions": nominal_optimizer_batch_positions,
                "planned_optimizer_steps": planned_optimizer_steps,
            },
            "precision": {
                "parameter_dtype": config.dtype,
                "autocast_dtype": config.autocast_dtype,
            },
            "data_manifest_sha256": self.manifest_sha256,
            "train_blocks": len(train_data),
            "validation_blocks": len(validation_data),
            "trainable_parameters": trainable,
            "trainable_pretrained_parameters": _parameter_count(trainable_pretrained),
            "trainable_added_parameters": _parameter_count(trainable_added),
            "added_parameters_total": _parameter_count(model.added_parameters()),
            "initialization_provenance": initialization_provenance,
        }

        is_resume_request = bool(config.resume_from or resume_auto)
        if self.run_info_path.exists():
            if not is_resume_request:
                raise FileExistsError(
                    f"output_dir already contains run.json; refuse fresh run in {self.run_dir}"
                )
        else:
            if resume_auto:
                occupied = any(
                    path.exists()
                    for path in (
                        self.metrics_path,
                        self.segments_path,
                        self.run_dir / "checkpoints",
                    )
                )
                if occupied:
                    raise RuntimeError(
                        "resume_auto found run artifacts but no run.json; refusing ambiguous recovery"
                    )
            _atomic_write_json(self.run_info_path, run_info)

        selected_checkpoint: Path | None = None
        fallback_used = False
        self._pending_validation_recovery = False
        if resume_auto and self.run_info_path.exists():
            try:
                selected_checkpoint, self.state, sampler_state, fallback_used = (
                    load_latest_valid_checkpoint(
                        self.run_dir,
                        model=self.model,
                        optimizer=self.optimizer,
                        expected_manifest_sha256=self.manifest_sha256,
                        expected_experiment_config=config.to_dict(),
                        expected_source_provenance=self.source,
                        allow_source_mismatch=self.allow_source_mismatch,
                        pass_scheduler=self.pass_scheduler,
                    )
                )
            except FileNotFoundError:
                if run_info_preexisting:
                    raise RuntimeError("existing run has no recoverable checkpoint")
            else:
                self._finish_resume(sampler_state)
        elif config.resume_from:
            selected_checkpoint = Path(config.resume_from)
            self.state, sampler_state = load_checkpoint(
                selected_checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                expected_manifest_sha256=self.manifest_sha256,
                expected_experiment_config=config.to_dict(),
                expected_source_provenance=self.source,
                allow_source_mismatch=self.allow_source_mismatch,
                pass_scheduler=self.pass_scheduler,
            )
            self._finish_resume(sampler_state)

        self.segment_id = _next_segment_id(self.segments_path)
        if selected_checkpoint is not None:
            repair = repair_metrics_to_checkpoint(self.metrics_path, self.state)
            metadata = inspect_checkpoint(selected_checkpoint)
            pending_validation = bool(
                metadata.get("checkpoint_metadata", {}).get("pending_validation")
            )
            self._pending_validation_recovery = (
                pending_validation and not self._validation_record_exists_at_current_state()
            )
            append_jsonl(
                self.metrics_path,
                {
                    "event": "resume",
                    "run_segment": self.segment_id,
                    "checkpoint": str(selected_checkpoint),
                    "fallback_used": bool(fallback_used),
                    "optimizer_steps": self.state.optimizer_steps,
                    "unique_tokens_seen": self.state.unique_tokens_seen,
                    "model_positions_seen": self.state.model_positions_seen,
                    "token_equivalent_compute": self.state.token_equivalent_compute,
                    "metrics_repair": repair,
                },
                durable=True,
            )
        append_jsonl(
            self.segments_path,
            {
                "event": "segment_start",
                "segment": self.segment_id,
                "parent_checkpoint": str(selected_checkpoint) if selected_checkpoint else None,
                "start_unique_tokens": self.state.unique_tokens_seen,
                "start_model_positions": self.state.model_positions_seen,
                "source": self.source,
                "hardware": self.hardware,
            },
            durable=True,
        )
        self._last_checkpoint_tokens = self.state.unique_tokens_seen
        self._last_checkpoint_time = time.monotonic()
        self._early_stop_satisfied = self._validation_stop_satisfied_at_current_state()

    def _finish_resume(self, sampler_state: dict) -> None:
        self._repair_optimizer_group_metadata()
        if self.state.phase != self.config.phase:
            raise ValueError(
                f"checkpoint phase {self.state.phase!r} does not match requested phase {self.config.phase!r}"
            )
        self.sampler.load_state_dict(sampler_state)

    def _validation_record_exists_at_current_state(self) -> bool:
        if not self.metrics_path.exists():
            return False
        with self.metrics_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("event") == "validation"
                    and int(record.get("optimizer_steps", -1)) == self.state.optimizer_steps
                    and int(record.get("unique_tokens_seen", -1)) == self.state.unique_tokens_seen
                    and int(record.get("model_positions_seen", -1)) == self.state.model_positions_seen
                ):
                    return True
        return False

    def _validation_stop_satisfied_at_current_state(self) -> bool:
        if self.config.early_stop is None or not self.metrics_path.exists():
            return False
        matching_record = None
        with self.metrics_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    record.get("event") == "validation"
                    and int(record.get("optimizer_steps", -1)) == self.state.optimizer_steps
                    and int(record.get("unique_tokens_seen", -1))
                    == self.state.unique_tokens_seen
                    and int(record.get("model_positions_seen", -1))
                    == self.state.model_positions_seen
                ):
                    matching_record = record
        if matching_record is None or "nll_by_pass" not in matching_record:
            return False
        return bool(
            _validation_stop_status(matching_record, self.config.early_stop)[
                "all_passed"
            ]
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        added_ids = {id(parameter) for parameter in self.model.added_parameters()}
        pretrained = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in added_ids
        ]
        added = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
            and id(parameter) in added_ids
        ]
        groups: list[dict] = []
        if pretrained:
            groups.append(
                {
                    "params": pretrained,
                    "lr": self.config.pretrained_lr,
                    "base_lr": self.config.pretrained_lr,
                    "group_name": "pretrained",
                    "weight_decay": self.config.weight_decay,
                }
            )
        if added:
            groups.append(
                {
                    "params": added,
                    "lr": self.config.added_lr,
                    "base_lr": self.config.added_lr,
                    "group_name": "added",
                    "weight_decay": self.config.weight_decay,
                }
            )
        if not groups:
            raise RuntimeError("no optimizer parameters")
        return torch.optim.AdamW(groups, foreach=False)

    def _repair_optimizer_group_metadata(self) -> None:
        added_ids = {id(parameter) for parameter in self.model.added_parameters()}

        def category(parameter) -> str:
            parameter_id = id(parameter)
            if parameter_id in added_ids:
                return "added"
            return "pretrained"

        base_lrs = {
            "pretrained": self.config.pretrained_lr,
            "added": self.config.added_lr,
        }
        for group in self.optimizer.param_groups:
            categories = {category(parameter) for parameter in group["params"]}
            if len(categories) != 1:
                raise RuntimeError("optimizer group mixes parameter categories")
            name = next(iter(categories))
            group.setdefault("group_name", name)
            group.setdefault("base_lr", base_lrs[name])

    def _set_lr(self) -> dict[str, float]:
        multiplier = lr_multiplier(
            self.state.unique_tokens_seen,
            total_tokens=self.config.max_unique_tokens,
            schedule=self.config.lr_schedule,
            legacy_warmup_tokens=self.config.warmup_tokens,
            legacy_min_lr_ratio=self.config.min_lr_ratio,
        )
        result = {"lr_multiplier": float(multiplier)}
        for group in self.optimizer.param_groups:
            base_lr = float(group["base_lr"])
            group["lr"] = base_lr * multiplier
            result[f"lr_{group['group_name']}"] = float(group["lr"])
        if "lr_pretrained" in result:
            result["lr"] = result["lr_pretrained"]
        elif "lr_added" in result:
            result["lr"] = result["lr_added"]
        return result

    def _checkpoint(self, *, pending_validation: bool = False) -> Path:
        path = save_checkpoint_generation(
            self.run_dir,
            model=self.model,
            optimizer=self.optimizer,
            sampler_state=self.sampler.state_dict(),
            pass_scheduler_state=self.pass_scheduler.state_dict(),
            train_state=self.state,
            experiment_config=self.config.to_dict(),
            data_manifest_sha256=self.manifest_sha256,
            source_provenance=self.source,
            checkpoint_metadata={"pending_validation": bool(pending_validation)},
            keep_last=self.config.checkpoint_keep_last,
        )
        self._last_checkpoint_tokens = self.state.unique_tokens_seen
        self._last_checkpoint_time = time.monotonic()
        return path

    def _checkpoint_due(self) -> bool:
        cfg = self.config
        token_due = bool(
            cfg.checkpoint_every_tokens
            and self.state.unique_tokens_seen - self._last_checkpoint_tokens
            >= cfg.checkpoint_every_tokens
        )
        time_due = bool(
            cfg.checkpoint_every_seconds
            and time.monotonic() - self._last_checkpoint_time
            >= cfg.checkpoint_every_seconds
        )
        return token_due or time_due

    def _pass_schedule_metrics(self) -> dict[str, object]:
        return {
            "pass_samples": self.pass_scheduler.samples,
            "pass_histogram": {
                str(passes): count
                for passes, count in sorted(self.pass_scheduler.histogram.items())
            },
        }

    def _evaluate(self) -> dict:
        with autocast_context(self.device, self.config.autocast_dtype):
            return self._evaluate_with_precision()

    def _evaluate_with_precision(self) -> dict:
        common = {
            "event": "validation",
            "run_segment": self.segment_id,
            "optimizer_steps": self.state.optimizer_steps,
            "unique_tokens_seen": self.state.unique_tokens_seen,
            "model_positions_seen": self.state.model_positions_seen,
            "control_positions_seen": self.state.model_positions_seen - self.state.unique_tokens_seen,
            "token_equivalent_compute": self.state.token_equivalent_compute,
            **self._pass_schedule_metrics(),
        }
        if self.config.eval_passes > 1:
            if not isinstance(self.model, MultiPassVariant):
                raise ValueError("eval_passes>1 requires a multipass variant")
            result = evaluate_pass_depth(
                self.model,
                self.validation_data,
                device=self.device,
                passes=self.config.eval_passes,
                max_blocks=self.config.eval_batches or None,
            )
            record = {
                **common,
                "nll": result.final_nll,
                "perplexity": result.final_perplexity,
                "predicted_tokens": result.predicted_tokens,
                "nll_by_pass": list(result.nll_by_pass),
                "perplexity_by_pass": list(result.perplexity_by_pass),
                "hidden_delta_rms": list(result.hidden_delta_rms),
                "nll_by_source_by_pass": list(result.nll_by_source_by_pass),
                "validation_blocks": result.blocks,
                "eval_passes": result.passes,
            }
        else:
            result = evaluate_nll(
                self.model,
                self.validation_data,
                device=self.device,
                max_blocks=self.config.eval_batches or None,
            )
            record = {
                **common,
                "nll": result.nll,
                "perplexity": result.perplexity,
                "predicted_tokens": result.predicted_tokens,
                "nll_by_source": result.nll_by_source,
                "validation_blocks": result.blocks,
                "eval_passes": result.passes,
            }
        if self.config.early_stop is not None:
            record["early_stop"] = _validation_stop_status(
                record, self.config.early_stop
            )
        append_jsonl(self.metrics_path, record)
        return record

    def _snapshot_crossed(self, previous_tokens: int) -> None:
        for threshold in self.config.snapshot_at_tokens or []:
            if not previous_tokens < threshold <= self.state.unique_tokens_seen:
                continue
            snapshot_dir = self.run_dir / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            actual = self.state.unique_tokens_seen
            path = snapshot_dir / f"model_{actual:012d}.safetensors"
            if path.exists():
                continue
            tensors = {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in self.model.state_dict().items()
            }
            temporary = path.with_name(path.name + ".tmp")
            save_safetensors(tensors, str(temporary))
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            metadata = {
                "requested_threshold": threshold,
                "unique_tokens_seen": actual,
                "model_positions_seen": self.state.model_positions_seen,
                "optimizer_steps": self.state.optimizer_steps,
                "source": self.source,
                "data_manifest_sha256": self.manifest_sha256,
                "variant": self.config.variant,
                "phase": self.config.phase,
            }
            _atomic_write_json(path.with_suffix(".json"), metadata)
            append_jsonl(
                self.metrics_path,
                {
                    "event": "snapshot",
                    "run_segment": self.segment_id,
                    "path": str(path),
                    "requested_threshold": threshold,
                    "unique_tokens_seen": actual,
                    "model_positions_seen": self.state.model_positions_seen,
                    "optimizer_steps": self.state.optimizer_steps,
                },
            )

    def _end_segment(self, reason: str) -> None:
        append_jsonl(
            self.segments_path,
            {
                "event": "segment_end",
                "segment": self.segment_id,
                "end_unique_tokens": self.state.unique_tokens_seen,
                "end_model_positions": self.state.model_positions_seen,
                "optimizer_steps": self.state.optimizer_steps,
                "reason": reason,
            },
            durable=True,
        )

    def train(self, *, until_unique_tokens: int | None = None) -> TrainState:
        cfg = self.config
        tokens_per_micro = cfg.batch_size * self.linguistic_per_block
        positions_per_micro = cfg.batch_size * self.train_data.sequence_length
        target_tokens = cfg.max_unique_tokens if until_unique_tokens is None else int(until_unique_tokens)
        if not self.state.unique_tokens_seen <= target_tokens <= cfg.max_unique_tokens:
            raise ValueError("until_unique_tokens must lie between current progress and max_unique_tokens")
        if target_tokens % tokens_per_micro:
            raise ValueError(
                "token budget must be divisible by batch_size * linguistic_tokens_per_block so the run ends exactly"
            )
        if self._early_stop_satisfied:
            self._end_segment("validation_gates")
            return self.state
        next_eval = (
            ((self.state.unique_tokens_seen // cfg.eval_every_tokens) + 1) * cfg.eval_every_tokens
            if cfg.eval_every_tokens else None
        )
        next_train_log = (
            ((self.state.unique_tokens_seen // cfg.train_log_every_tokens) + 1)
            * cfg.train_log_every_tokens
            if cfg.train_log_every_tokens else None
        )
        log_window_start = self.state.unique_tokens_seen
        log_window_updates = 0
        log_window_sums: dict[str, float] = defaultdict(float)
        log_window_counts: dict[str, int] = defaultdict(int)
        log_window_elapsed_seconds = 0.0
        log_window_tokens = 0
        log_window_positions = 0
        log_window_histogram_start = dict(self.pass_scheduler.histogram)

        def flush_train_log(last_record: dict, *, partial: bool) -> None:
            nonlocal log_window_start
            nonlocal log_window_updates
            nonlocal log_window_elapsed_seconds
            nonlocal log_window_tokens
            nonlocal log_window_positions
            nonlocal log_window_histogram_start
            if log_window_updates == 0:
                return
            logged = dict(last_record)
            logged["log_interval_start_tokens"] = log_window_start
            logged["log_interval_tokens"] = (
                self.state.unique_tokens_seen - log_window_start
            )
            logged["log_interval_updates"] = log_window_updates
            logged["log_interval_elapsed_seconds"] = log_window_elapsed_seconds
            logged["log_interval_partial"] = partial
            for key, value in log_window_sums.items():
                logged[key] = value / log_window_counts[key]
            logged["metric_observation_counts"] = dict(
                sorted(log_window_counts.items())
            )
            logged["tokens_per_second"] = (
                log_window_tokens / log_window_elapsed_seconds
            )
            logged["model_positions_per_second"] = (
                log_window_positions / log_window_elapsed_seconds
            )
            interval_histogram = {
                str(passes): count - log_window_histogram_start.get(passes, 0)
                for passes, count in sorted(self.pass_scheduler.histogram.items())
                if count - log_window_histogram_start.get(passes, 0)
            }
            logged["pass_histogram_interval"] = interval_histogram
            logged["pass_samples_interval"] = sum(interval_histogram.values())
            append_jsonl(self.metrics_path, logged)
            log_window_start = self.state.unique_tokens_seen
            log_window_updates = 0
            log_window_sums.clear()
            log_window_counts.clear()
            log_window_elapsed_seconds = 0.0
            log_window_tokens = 0
            log_window_positions = 0
            log_window_histogram_start = dict(self.pass_scheduler.histogram)
        if self._pending_validation_recovery:
            validation = self._evaluate()
            self._pending_validation_recovery = False
            self._early_stop_satisfied = bool(
                validation.get("early_stop", {}).get("all_passed", False)
            )
            if self._early_stop_satisfied:
                self._end_segment("validation_gates")
                return self.state

        self.model.train()
        while self.state.unique_tokens_seen < target_tokens:
            previous_tokens = self.state.unique_tokens_seen
            start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            update_passes = 0
            update_metrics: dict[str, float] = defaultdict(float)
            update_metric_counts: dict[str, int] = defaultdict(int)
            remaining_micro = (target_tokens - self.state.unique_tokens_seen) // tokens_per_micro
            accumulation_steps = min(cfg.grad_accum_steps, remaining_micro)
            if accumulation_steps <= 0:
                raise RuntimeError("invalid zero-length optimizer update")

            for _ in range(accumulation_steps):
                indices = self.sampler.next_indices(cfg.batch_size)
                ids = self.train_data.batch(indices, device=self.device)
                if int(ids.numel()) != positions_per_micro:
                    raise RuntimeError("physical packed sequence length changed across microbatches")
                passes = self.pass_scheduler.sample(self.state.unique_tokens_seen)
                with autocast_context(self.device, cfg.autocast_dtype):
                    output = self.model.compute_loss(
                        ids,
                        phase=cfg.phase,
                        passes=passes,
                        loss_weights=cfg.ntp_loss_weights_for_passes(passes),
                    )
                if not bool(torch.isfinite(output.loss).item()):
                    raise RuntimeError("non-finite training loss")
                (output.loss / accumulation_steps).backward()
                update_loss += float(output.loss.detach().cpu())
                update_passes += output.effective_passes
                for key, value in output.metrics.items():
                    update_metrics[key] += float(value)
                    update_metric_counts[key] += 1
                self.state.micro_steps += 1
                self.state.unique_tokens_seen += tokens_per_micro
                self.state.model_positions_seen += positions_per_micro
                self.state.token_equivalent_compute += positions_per_micro * output.effective_passes

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("non-finite gradient norm")
            lr_record = self._set_lr()
            self.optimizer.step()
            synchronize(self.device)
            self.state.optimizer_steps += 1
            elapsed = max(time.perf_counter() - start, 1e-9)
            record = {
                "event": "train",
                "run_segment": self.segment_id,
                "phase": cfg.phase,
                "optimizer_steps": self.state.optimizer_steps,
                "micro_steps": self.state.micro_steps,
                "unique_tokens_seen": self.state.unique_tokens_seen,
                "model_positions_seen": self.state.model_positions_seen,
                "control_positions_seen": self.state.model_positions_seen - self.state.unique_tokens_seen,
                "token_equivalent_compute": self.state.token_equivalent_compute,
                "loss": update_loss / accumulation_steps,
                "grad_norm": float(grad_norm.detach().cpu()),
                "tokens_per_second": (tokens_per_micro * accumulation_steps) / elapsed,
                "model_positions_per_second": (positions_per_micro * accumulation_steps) / elapsed,
                "microbatch_tokens": tokens_per_micro,
                "microbatch_model_positions": positions_per_micro,
                "accumulation_steps": accumulation_steps,
                "optimizer_batch_tokens": tokens_per_micro * accumulation_steps,
                "optimizer_batch_model_positions": positions_per_micro * accumulation_steps,
                "nominal_optimizer_batch_tokens": tokens_per_micro * cfg.grad_accum_steps,
                "nominal_optimizer_batch_model_positions": positions_per_micro * cfg.grad_accum_steps,
                "mean_passes": update_passes / accumulation_steps,
                **self._pass_schedule_metrics(),
                **lr_record,
            }
            record.update(
                {
                    key: value / update_metric_counts[key]
                    for key, value in sorted(update_metrics.items())
                }
            )
            record["metric_observation_counts"] = dict(
                sorted(update_metric_counts.items())
            )
            if cfg.train_log_every_tokens:
                log_window_updates += 1
                for key in ("loss", "grad_norm", "mean_passes"):
                    log_window_sums[key] += float(record[key])
                    log_window_counts[key] += 1
                for key, count in update_metric_counts.items():
                    log_window_sums[key] += float(record[key]) * count
                    log_window_counts[key] += count
                log_window_elapsed_seconds += elapsed
                log_window_tokens += int(record["optimizer_batch_tokens"])
                log_window_positions += int(
                    record["optimizer_batch_model_positions"]
                )

            log_due = next_train_log is not None and self.state.unique_tokens_seen >= next_train_log
            final_update = self.state.unique_tokens_seen >= target_tokens
            if cfg.train_log_every_tokens == 0 or log_due or final_update:
                if cfg.train_log_every_tokens:
                    flush_train_log(record, partial=False)
                else:
                    append_jsonl(self.metrics_path, record)
                if next_train_log is not None:
                    while next_train_log <= self.state.unique_tokens_seen:
                        next_train_log += cfg.train_log_every_tokens
            self._snapshot_crossed(previous_tokens)

            eval_due = next_eval is not None and self.state.unique_tokens_seen >= next_eval
            # Save trained state before an expensive read-only evaluation. A
            # hard VM loss inside evaluation therefore loses no optimizer work.
            if self._checkpoint_due() or (
                eval_due and self.state.unique_tokens_seen > self._last_checkpoint_tokens
            ):
                self._checkpoint(pending_validation=bool(eval_due))

            if eval_due:
                validation = self._evaluate()
                while next_eval is not None and next_eval <= self.state.unique_tokens_seen:
                    next_eval += cfg.eval_every_tokens
                self._early_stop_satisfied = bool(
                    validation.get("early_stop", {}).get("all_passed", False)
                )
                if self._early_stop_satisfied:
                    if cfg.train_log_every_tokens:
                        flush_train_log(record, partial=True)
                    self._end_segment("validation_gates")
                    return self.state
                self.model.train()

            if self.stop_requested():
                if cfg.train_log_every_tokens:
                    flush_train_log(record, partial=True)
                if self.state.unique_tokens_seen > self._last_checkpoint_tokens:
                    self._checkpoint()
                self._end_segment("signal")
                return self.state

        if self.state.unique_tokens_seen > self._last_checkpoint_tokens:
            self._checkpoint()
        final_validation = self._evaluate() if cfg.eval_batches else None
        self._early_stop_satisfied = bool(
            final_validation
            and final_validation.get("early_stop", {}).get("all_passed", False)
        )
        self._end_segment(
            "validation_gates" if self._early_stop_satisfied else "completed"
        )
        return self.state
