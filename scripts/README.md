# Scripts

Scripts are thin command-line entry points. Data recipes live under `data/`,
evaluation suites under `evaluation/`, and experiment settings with their owning
benchmark study/control.

## Setup and validation

```text
download_model.py
verify_model.py
compare_to_hf.py
compare_to_hf_layers.py
compare_to_hf_inputs_embeds.py
prepare_data.py
verify_data.py
verify_data_disjointness.py
verify_study.py
smoke_mps.py
```

## Efficiency and cloud qualification

```text
benchmark_training_efficiency.py
benchmark_inference_efficiency.py
estimate_training_flops.py
cloud_preflight.py
```

The efficiency runner performs real optimizer steps and reports linguistic-token
and physical-position throughput when they differ. Memory Attention cases must
state their write policy explicitly. The paper-policy `forward_modes.yaml`
suite and its Makefile targets are retired. The general training/precision/
scaling suites remain; they are optional engineering measurements, not new
scientific arms or qualification of unmeasured merger architectures.

`benchmark_inference_efficiency.py` measures full-block K-pass validation and
cached standard, feedback, exact, and diagnostic continuation costs on CUDA.
Its synthetic timing protocol is documented in
`benchmarks/development/inference_efficiency/README.md`.

`cloud_preflight.py` checks CUDA/model/data/source/run compatibility, persistent
storage, free space, and memory-token-expanded batching before a paid run.

`estimate_training_flops.py --study <STUDY.yaml>` derives one dominant-matmul
estimate per arm directly from its authoritative experiment config and data
recipe. The report identifies its conventional backward and frozen-parameter
limitations; it is an algorithmic estimate, not measured accelerator work.

## Training and evaluation

```text
train.py
run_study.py
evaluate_nll.py
evaluate_lm_harness.py
evaluate_pass_depth.py
evaluate_memory_interventions.py
evaluate_recurrent_inference.py
evaluate_parameter_drift.py
generate.py
```

`run_study.py` is the common executor for colocated development/core studies.
It validates the manifest, exercises every sampled pass depth with a
forward/backward preflight at the config's declared physical batch, and then
runs selected arms sequentially. Paper replay/BPTT execution is removed.

`start-and-watch` is the unattended cloud wrapper. It starts a remote
`train.py --resume-auto` process, waits for a durable completed segment,
transfers the output, verifies SHA-256 hashes, and then shuts down or deletes
the Verda compute instance. It leaves the VM untouched when a run is
interrupted or transfer verification fails. Run `--help` for the full
interface; use `--transfer metadata` only for small smoke checks.

`run-cloud-study` applies that lifecycle to the manifest arms of any explicitly
selected locked study. It skips locally complete arms, transfers full artifacts,
deletes each verified remote run directory, shuts down between arms, and deletes
the compute instance after the selected study. A lock file prevents two study
controllers from using the same VM concurrently.

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_comparison \
  --wire-only --wire-device cuda
```

Training/evaluation loaders automatically wrap ordinary packed artifacts with
`MemoryTokenPackedDataset` when `memory_write_mode: memory_token`. The stored
data remain ordinary linguistic IDs; the view inserts input-only control ID V at
load time.

Evaluation commands require either `--checkpoint` or the explicit
`--initialized-baseline` time-zero mode. `evaluate_nll.py` reports the final
parallel pass; `evaluate_pass_depth.py` reports every pass through the experiment's
`eval_passes` (or an explicit `--passes` override). Both accept an independent
`--evaluation-data-dir`.

`evaluate_lm_harness.py` accepts independent `--prefill-passes K` and
`--decode-mode standard|feedback` overrides, with experiment defaults described
below. Candidate suites retain answer scores and
margins; generation suites retain generated samples. `evaluate_parameter_drift.py`
separates backbone and added-module movement from an architecture-compatible
reference. `verify_data_disjointness.py` rejects shared complete tokenized
documents between the evaluation split and each training/wiring artifact.

Pass-depth, memory interventions, and exact-vs-feedback continuation scripts
are reusable checkpoint diagnostics.
`evaluate_memory_interventions.py` measures one feedback transition and can
independently intervene on the historical Recirculation–Memory Attention hybrid's
recurrent source and slow memory source. It requires at least two validation blocks for a
genuine mismatch condition.

Public `generate.py`/model generation remain ordinary language generation. The
low-level recurrent API can consume explicit MEM control steps, but no sampler
silently schedules architecture control positions.

## Evaluation defaults

The evaluation commands now share precision and scoring. Parallel depth defaults
to `eval_passes`; downstream prompt depth defaults to `eval_prefill_passes` or
`eval_passes`, independently of `eval_decode_mode`. Explicit flags override these
values. `--autocast-dtype config|float32|bfloat16` records actual evaluation
precision without changing checkpoint compatibility settings. Standalone
`--max-blocks` is a prefix limit; omit it for the full split, or use 64 to match
the active routine check. See [the evaluation contract](../evaluation/README.md).

Snapshot publication and recovery are documented in
[docs/TRAINING.md](../docs/TRAINING.md). Use `evaluate_nll.py --forward feedback
--max-blocks 1` for full-block BOS-only feedback; it fixes prefill to K=1 and
reports full and aligned target scores. The trainer can run the same evaluator
at selected `feedback_eval_at_tokens` snapshot thresholds. This adds no new
inference algorithm and does not change routine K=4 checks.
