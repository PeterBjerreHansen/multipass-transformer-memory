# CUDA / cloud execution

Cloud execution is provider-agnostic. A serious run assumes a Linux CUDA host,
a persistent filesystem, the pinned model/data inputs, and this repository's
locked Python environment.

## Check the active protocol and hardware fit

The active frozen studies use 2048-token blocks with microbatch 8 and
accumulation 4. Hardware tuning is optional: preserve 32 sequences per optimizer
update and record the selected pair. The old 1024-token paper-policy suite and
its Makefile targets are retired; paper replay/BPTT execution is deleted.

On the intended GPU, use the existing checks and optional engineering suites:

```bash
uv sync --extra data --extra eval
make check
# Optional training-cost characterization; not a scientific arm:
make efficiency-cuda-training
```

The general engineering grid includes historical shifted recirculation; it does
not certify the two new recurrent mergers. Use the actual arm config for a
forward/backward preflight. Memory-token benchmarks distinguish linguistic
length from physical model positions.

[A6000 measurements](../benchmarks/development/inference_efficiency/README.md)
support routine K=4 checks at the existing cadence. Optional feedback validation
runs only at selected durable snapshots, initially one full block per arm. It
pauses training outside measured optimizer time, not at every routine check.
Evaluation inherits and records experiment precision;
use an explicit override for a matched FP32 diagnostic. Pending snapshot work
recovers before training advances; see [TRAINING.md](TRAINING.md).

## Paid-run preflight

Use the same config and persistent volume that the training process will use:

```bash
uv run python scripts/cloud_preflight.py \
  --config <config.yaml> \
  --mode auto \
  --persistent-root /mnt/<persistent-volume>
```

`--mode auto` resolves an empty output directory as a new run and a compatible
existing run as resume. Existing ambiguous trajectory artifacts fail closed.
`--mode new` and `--mode resume` are available when operator intent should be
explicit.

The preflight checks:

- CUDA/BF16 capability;
- the pinned TinyMistral checkpoint and data artifact integrity;
- source cleanliness and execution-code/`uv.lock` provenance;
- run/checkpoint compatibility and recoverability;
- output directory containment beneath the persistent root;
- free disk space for durable checkpoint rotation;
- linguistic versus physical batching quantities for memory-token runs;
- the exact Memory Attention or Strided Attention configuration.

## Spot-safe checkpoint policy

A serious spot run should retain two complete resumable generations and save on
both a token and wall-clock cadence, for example:

```yaml
checkpoint_every_tokens: 500000
checkpoint_every_seconds: 600
checkpoint_keep_last: 2
```

The cadence should be measured on the selected GPU/storage pair. The goal is to
bound lost compute while keeping checkpoint I/O a small fraction of wall time.

Generation state is:

```text
<output_dir>/checkpoints/
  checkpoint_00012001280.pt
  checkpoint_00012500992.pt
  latest.json
```

A new checkpoint is written to a temporary file, flushed/fsynced, atomically
renamed, reopened for validation, and only then advertised by `latest.json`.
The oldest generation is pruned only after the new pointer is durable. Auto
resume tries the current generation and falls back to the previous valid one if
the newest is unreadable.

The checkpoint is the trajectory source of truth. On resume, `metrics.jsonl` is
repaired back to the selected checkpoint so metric rows written after durable
training progress are discarded.

## Run provenance

`run.json` is the immutable experiment description. `segments.jsonl` records
each process lifetime, parent checkpoint, source identity, and hardware. Resume
checks deterministic execution-code and `uv.lock` hashes; Git metadata is useful
when present but an identical source archive can still be identified by content
hash.

`--allow-source-mismatch` is a development escape hatch and should not be used
for a locked campaign.

## Training invocation

Use the same command on the first and replacement instances:

```bash
uv run python scripts/train.py \
  --config <config.yaml> \
  --resume-auto
```

If the run directory is truly new, this starts from zero. Otherwise it resumes
the newest valid generation. A pre-existing run without a recoverable
checkpoint is a hard failure; auto mode never silently restarts it.

SIGINT/SIGTERM request a checkpoint at the next completed optimizer boundary.
Correctness does not depend on receiving a signal: the hard failure model is an
instance that disappears without executing another instruction.

## Scientific snapshots

Resumable checkpoints are operational and only the newest configured generations
are retained. Optional `snapshot_at_tokens` writes weights-only safetensors to
`<output_dir>/snapshots/` for analysis. They are never used by auto-resume.

The fresh unfrozen comparison must not initialize from the frozen trajectory.
Pin and verify its training/evaluation artifacts explicitly. Keep the entire run
directory on persistent storage, including snapshots and `evaluations/` reports,
and back it up separately from an ephemeral instance. New snapshots carry
weights and identity together; resume drains pending publication/selected
feedback work before training advances. See [TRAINING.md](TRAINING.md).

## Unattended start, transfer, and cleanup

`scripts/start-and-watch` runs the training process in the VM, watches the
durable `segments.jsonl` journal, transfers the completed output, verifies a
SHA-256 manifest, and only then performs the requested VM cleanup. It does not
estimate completion from elapsed time. If the process exits without a completed
segment, it fails closed and leaves the VM available for recovery.

The default cleanup is `shutdown`. `--cleanup delete` deletes only the compute
instance; it deliberately leaves attached volumes in place. Use
`--transfer metadata` only for a smoke check or when large checkpoints and
weights have already been archived elsewhere. Serious runs should use the
default `--transfer all`.

Example for a serious run:

```bash
./scripts/start-and-watch \
  --host <vm-ip> \
  --vm-id <verda-vm-id> \
  --config benchmarks/core/<locked-study>/<arm>.yaml \
  --remote-output benchmarks/core/<locked-study>/results/<arm> \
  --local-output benchmarks/core/<locked-study>/results/<arm> \
  --start-vm \
  --cleanup delete
```

The local computer must remain powered on while the controller is running.
To detach it from the active Codex turn, launch it from a terminal with
`nohup` and keep its log locally:

```bash
nohup ./scripts/start-and-watch <same-options-as-above> \
  > /tmp/tinymistral-start-and-watch.log 2>&1 &
echo $!
```

The script sends a macOS notification on success or failure when
`osascript` is available. It continues independently of Codex and does not
consume model tokens while waiting between status checks.

For a locked multi-arm study, use `scripts/run-cloud-study`. It reads the arm
IDs and config names from that study's manifest, applies the same transfer and
checksum boundary to each selected arm, removes each verified remote run
directory before shutdown, and deletes the compute instance after all arms
complete. It is restartable because locally complete arms are skipped:

```bash
nohup caffeinate -dimsu ./scripts/run-cloud-study \
  --host <vm-ip> \
  --vm-id <verda-vm-id> \
  --study-dir benchmarks/core/<locked-study> \
  > /tmp/tinymistral-cloud-study.log 2>&1 &
echo $!
```

The wrapper keeps the persistent OS volume by omitting `--with-volumes` from
the final Verda delete operation. On failure it shuts down the compute
instance but retains remote artifacts for recovery; it does not delete a
failed run's data.

Use `--arm` to run only selected manifest arms. Locally complete arms are
skipped, so the same command is safe to resume. The wrapper rejects studies
whose manifest is not locked.
