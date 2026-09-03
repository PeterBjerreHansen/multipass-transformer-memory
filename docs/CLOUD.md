# CUDA and cloud operation

The trainer and input preflight are provider-agnostic.
The unattended lifecycle wrappers below use the Verda CLI.
Use a Linux CUDA host, persistent storage and the locked Python environment.

## Pre-training checks

Run these checks before the frozen LR sweep. Preserve the study's effective optimizer batch.
If microbatch size changes, adjust accumulation and record the resolved config.

1. Stage the pinned model and packed dataset on the target host.
2. Install dependencies with `uv sync --extra data --extra eval`.
3. Run `make check`.
4. Check the actual run config and storage:

   ```bash
   uv run python scripts/cloud_preflight.py \
     --config <arm.yaml> --mode auto \
     --persistent-root /mnt/<persistent-volume>
   ```

5. Check all declared training depths:

   ```bash
   uv run python scripts/run_study.py \
     --study-dir benchmarks/development/frozen_backbone_comparison \
     --wire-only --wire-device cuda
   ```

6. Run a bounded real-trainer check in a separate preflight output directory.
   Exercise actual optimizer updates, including K=3 and optimizer-state allocation.
   Run routine K=4 validation and the production BOS evaluator on one complete block.
   Exercise snapshot publication, interruption and resume.
   Use short preflight-only thresholds without changing scientific milestones.

The final step is an operator task, not a new automated preflight command.
Do not initialize scientific runs from preflight weights.
Dedicated initial-baseline evaluation and expanded memory diagnostics are not prerequisites.

`cloud_preflight.py` checks CUDA/BF16 capability, pinned inputs, source cleanliness,
run compatibility, output containment and estimated checkpoint-rotation space.
It does not measure training VRAM or run an optimizer update.
Its default mode is `new`. Explicit `auto` resolves a new output versus a recoverable run.
Ambiguous existing state fails closed. `--mode resume` requires a recoverable checkpoint.
An explicit persistent root enables containment and free-space checks.
Create the output parent directory before the disk-space check.
Budget separately for datasets, snapshots, reports and caches.

`--wire-only` performs forward/backward checks without an optimizer step.
The general [engineering suites](../benchmarks/efficiency/README.md) use synthetic inputs
and do not qualify both active recurrent mergers.
Neither cloud wrapper automatically runs the complete pre-training checklist.

## Persistent state and resume

Keep the entire output directory on persistent storage and back it up separately.
This includes checkpoints, snapshots, journals and evaluation reports.
Two rolling checkpoint generations allow fallback if the newest becomes unreadable.

Use the same invocation on the initial and replacement hosts:

```bash
uv run python scripts/train.py --config <arm.yaml> --resume-auto
```

Auto-resume starts fresh only when no trajectory artifacts exist.
Otherwise it restores the newest valid checkpoint or fails without restarting.
SIGINT/SIGTERM request a checkpoint at an optimizer boundary.
Periodic checkpoints limit lost work when a host disappears without warning.

`run.json` records the experiment. `segments.jsonl` records process lifetimes.
Resume checks execution-code and lockfile hashes as well as compatible settings.
`--allow-source-mismatch` is a development override, not a locked-campaign default.
See [training](TRAINING.md) for checkpoint publication, metrics repair,
snapshot retention and pending feedback recovery.
See [evaluation](../evaluation/README.md#precision) for precision choices.

## One-run controller

`scripts/start-and-watch` starts or observes remote training.
It waits for a durable completed segment, transfers outputs and checks SHA-256 hashes.
Only then does it apply the requested VM cleanup.

- `--cleanup shutdown` is the default.
- `--cleanup delete` deletes the compute instance but retains attached volumes.
- `--cleanup none` retains the running instance.
- `--transfer all` is the default and includes checkpoints and snapshots.
- Use `--transfer metadata` only when large artifacts are archived elsewhere or unnecessary for a smoke check.
- `--delete-remote-output` removes the verified remote run directory after transfer.
- `--watch-existing` permits transfer of an already completed run.

A failed training process or transfer leaves the one-run controller's VM and artifacts available for recovery.
Inspect them before retrying.

```bash
./scripts/start-and-watch \
  --host <vm-ip> --vm-id <verda-vm-id> \
  --config benchmarks/core/<locked-study>/<arm>.yaml \
  --remote-output benchmarks/core/<locked-study>/results/<arm> \
  --local-output benchmarks/core/<locked-study>/results/<arm> \
  --start-vm --cleanup delete
```

The local controller must remain running. Use a terminal process when it must outlive an interactive session.
The wrapper sends a macOS notification when `osascript` is available, unless `--no-notify` is set.

## Locked-study controller

`scripts/run-cloud-study` reads selected arm IDs from a verified locked manifest.
The current planned studies are not eligible until reviewed and locked.

```bash
nohup caffeinate -dimsu ./scripts/run-cloud-study \
  --host <vm-ip> --vm-id <verda-vm-id> \
  --study-dir benchmarks/core/<locked-study> \
  > /tmp/tinymistral-cloud-study.log 2>&1 &
```

The wrapper transfers all artifacts and verifies each local result.
It removes each verified remote run directory and shuts down between arms.
After all selected arms complete, it deletes compute without deleting attached volumes.
Use `--keep-vm` to retain the instance instead.
A lock prevents concurrent local controllers from claiming the same configured lock file.

On failure, this wrapper attempts VM shutdown but retains remote artifacts.
This differs from the one-run controller's failure behavior.
Locally complete, checksum-verified arms are skipped on restart.
Use repeated `--arm` options for a subset.
