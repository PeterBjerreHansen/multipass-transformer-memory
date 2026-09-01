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
estimate_training_flops.py
cloud_preflight.py
```

The efficiency runner performs real optimizer steps and reports linguistic-token
and physical-position throughput when they differ. Memory Attention cases must
state their write policy explicitly. The active `forward_modes.yaml` suite
qualifies full BPTT, candidate TBPTT windows, and whole-block multipass training
on the target GPU.

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
It validates the manifest, exercises every sampled pass depth with a one-batch
forward/backward preflight, and then runs selected arms sequentially.

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
  --study-dir benchmarks/development/forward_policy_qualification \
  --wire-only --wire-device cuda
```

Training/evaluation loaders automatically wrap ordinary packed artifacts with
`MemoryTokenPackedDataset` when `memory_write_mode: memory_token`. The stored
data remain ordinary linguistic IDs; the view inserts input-only control ID V at
load time.

Evaluation commands require either `--checkpoint` or the explicit
`--initialized-baseline` time-zero mode. `evaluate_nll.py` requires one pass
depth. `evaluate_pass_depth.py` reports exact full-sequence K=1 through K=8 by
default. Both accept an independent `--evaluation-data-dir`.

`evaluate_lm_harness.py` requires prompt `--prefill-passes K` and an independent
`--decode-mode standard|feedback`. Candidate suites retain answer scores and
margins; generation suites retain generated samples. `evaluate_parameter_drift.py`
separates backbone and added-module movement from an architecture-compatible
reference. `verify_data_disjointness.py` rejects shared complete tokenized
documents between the evaluation split and each training/wiring artifact.

Pass-depth, memory interventions, and exact-vs-feedback continuation scripts
are reusable checkpoint diagnostics.
`evaluate_memory_interventions.py` measures one feedback transition and can
independently intervene on the active Recirculation–Memory Attention hybrid's
recurrent source and slow memory source. It requires at least two validation blocks for a
genuine mismatch condition.

Public `generate.py`/model generation remain ordinary language generation. The
low-level recurrent API can consume explicit MEM control steps, but no sampler
silently schedules architecture control positions.
