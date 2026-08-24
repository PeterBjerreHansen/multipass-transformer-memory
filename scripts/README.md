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
verify_study.py
smoke_mps.py
```

## Efficiency and cloud qualification

```text
benchmark_training_efficiency.py
select_cuda_batch.py
cloud_preflight.py
```

The efficiency runner performs real optimizer steps and reports linguistic-token
and physical-position throughput when they differ. Memory Attention cases must state their
write policy explicitly. `select_cuda_batch.py` consumes the dedicated CUDA K=2
qualification and chooses the smallest common efficient adaptive-Recirculation/
dense-Memory-Attention microbatch rather than assuming maximum feasible batch is
scientifically valid.

`cloud_preflight.py` checks CUDA/model/data/source/run compatibility, persistent
storage, free space, and memory-token-expanded batching before a paid run.

## Training and evaluation

```text
train.py
run_study.py
evaluate_nll.py
evaluate_lm_harness.py
evaluate_pass_depth.py
evaluate_memory_interventions.py
evaluate_recurrent_inference.py
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

`run-cloud-campaign` applies that lifecycle to selected arms of the locked
Stage-5 eight-arm study. It can be used for a complete campaign or to resume
an interrupted subset; locally complete arms are skipped.
It skips locally complete arms, transfers full artifacts, deletes
each verified remote run directory, shuts down between arms, and deletes the
compute instance after the selected campaign. A lock file prevents two
campaigns from using the same VM concurrently.

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_1_wiring \
  --wire-only --wire-device mps
```

Training/evaluation loaders automatically wrap ordinary packed artifacts with
`MemoryTokenPackedDataset` when `memory_write_mode: memory_token`. The stored
data remain ordinary linguistic IDs; the view inserts input-only control ID V at
load time.

`evaluate_nll.py` reports pass-1 NLL by default; use `--passes K` to make the
pass depth explicit for a multipass model. Pass-depth, memory interventions,
and recurrent-inference scripts are reusable checkpoint diagnostics.
`evaluate_memory_interventions.py` measures one feedback transition and can
independently intervene on the active Recirculation–Memory Attention hybrid's
recurrent source and slow memory source. It requires at least two validation blocks for a
genuine mismatch condition.

Public `generate.py`/model generation remain ordinary language generation. The
low-level recurrent API can consume explicit MEM control steps, but no sampler
silently schedules architecture control positions.
