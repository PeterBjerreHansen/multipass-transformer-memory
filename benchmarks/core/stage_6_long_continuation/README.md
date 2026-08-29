# Stage 6: long plastic continuation

This study tests whether a substantially more plastic backbone update can
produce capability movement when the Dense SWA Memory Attention model is given
billions, rather than millions, of continuation tokens.

The base model remains the pinned final `M4-ai/TinyMistral-248M-v3` checkpoint
(about 21B upstream training tokens). The earlier ~17B upstream checkpoint was
screened as a possible higher-headroom start, but is intentionally not part of
this study so that the existing 100M comparison remains interpretable.

## Locked protocol

- **Starting weights:** the canonical pinned `M4-ai/TinyMistral-248M-v3`
  backbone (`src/tiny_mistral/loading.py` records the immutable revision), plus
  the canonical 5M-token Dense SWA Memory Attention wiring checkpoint at
  `benchmarks/development/stage_1_wiring/results/bank_dense_wiring_5m/checkpoints/checkpoint_000005242880.pt`.
  `init_from` loads weights only; the new run starts a fresh optimizer and
  scheduler trajectory.
- **Data:** `data/dolmino/gpu_2048_long_2p5b`, prepared from the pinned DOLMino
  revision with the same 5,242,880-token wiring offset. The artifact contains
  2,499,999,744 continuation tokens and the shared 2,000,896-token validation
  split. Materialize it before launching the arm:

  ```bash
  uv run python scripts/prepare_data.py \
    --config data/dolmino/gpu_2048_long_2p5b/config.yaml
  uv run python scripts/verify_data.py \
    data/dolmino/gpu_2048_long_2p5b
  ```

  The first 100M-token region is the same post-wiring source mixture used by
  Stage 5, while the remainder extends beyond the old 100M artifact. The larger
  artifact has its own deterministic block permutation, so the 100M comparison
  is distribution- and budget-matched rather than bit-for-bit batch-order
  matched.
- **Architecture:** Dense SWA Memory Attention, with K=2/K=3 sampled as
  `90% / 10%`. Within a sampled example, the final pass carries 90% of the NTP
  loss and pass 1 carries 10%.
- **Optimization:** backbone peak LR `1e-5`, added-module peak LR `3e-5`,
  token-based cosine decay over the complete 2.5B-token run, 5,242,880-token
  warmup, and a final multiplier of `0.1`. There is no scientific early-stop
  gate; monitoring is observational, with only numerical failure treated as a
  reason to interrupt the run.
  The backbone rate is the selected Dense rate from the completed local LR
  sweep; that sweep held the added-module rate at `3e-5`, which remains fixed
  here so the long run changes only the continuation horizon and backbone
  plasticity.
- **Paired vanilla control:** `vanilla_2p5b.yaml` uses the same data artifact,
  seeds, 2.5B-token cosine horizon, warmup, and backbone LR, but removes the
  memory module and uses one pass. It has its own output directory and volume
  so it can run concurrently without sharing the block volume.
- **Hardware contract:** CUDA BF16 autocast with FP32 parameters and optimizer,
  batch size 1, and one optimizer update per 2,048 linguistic tokens. The
  target is approximately 2.5B fresh tokens in the available A6000 budget.

## Cloud storage and memory planning

The generated `gpu_2048_long_2p5b` artifact is approximately 5.0 GB on disk:
the packed training tokens use 5.0 GB, while validation, source IDs, and the
manifest add only a few MB. Materialization temporarily keeps the per-source
files alongside the final packed files, so data preparation needs about 10 GB
of free space before software caches are considered.

Using the measured Stage-5 artifact sizes as a planning estimate, this run also
needs approximately 1.0 GB for the initialization checkpoint, 6.1 GB for the
two retained training checkpoints, and 5.1 GB for the five milestone snapshots.
The long-run journal aggregates training updates over 4,194,304-token windows,
so it should remain in the MB range rather than growing by one record per
optimizer step. Validation runs every 8,388,608 tokens. Checkpoints are
bounded by 16,777,216 tokens and 15 minutes, while the five milestone
snapshots remain mandatory. The resulting retained run artifacts are about
18–19 GB before the OS image, CUDA/`uv` environment, dataset metadata caches,
and safety margin.

The single-A6000 deployment has 48 GB of GPU memory and 60 GB of host RAM. The
dataset is streamed and memory-mapped, so its size is primarily a persistent
storage concern rather than a RAM requirement. CUDA preflight must still pass
on the selected instance before launch.

Use a dedicated persistent NVMe volume of at least **100 GiB** for the
repository, materialized data, and run output; **150 GiB** is more comfortable
if caches or additional artifacts are retained. Do not copy the large Stage-5
results collection to the VM. Mount the volume before data preparation and run
`scripts/cloud_preflight.py` with that mount as `--persistent-root`.

## Launch checklist

The checked-in recipe is intentionally small; the generated 2.5B-token data
artifact is not tracked. On the persistent GPU volume, run these commands from
the repository root before starting training:

```bash
uv run python scripts/prepare_data.py \
  --config data/dolmino/gpu_2048_long_2p5b/config.yaml
uv run python scripts/verify_data.py \
  data/dolmino/gpu_2048_long_2p5b
uv run python scripts/cloud_preflight.py \
  --config benchmarks/core/stage_6_long_continuation/dense_swa_memory_attention_plastic_2p5b.yaml \
  --persistent-root /mnt/<persistent-volume> \
  --mode new
uv run python scripts/run_study.py \
  --study-dir benchmarks/core/stage_6_long_continuation \
  --arm dense_swa_memory_attention_plastic_2p5b
```

`cloud_preflight.py` deliberately refuses a dirty checkout unless
`--allow-dirty` is supplied. After an interruption, rerun the same
`run_study.py` command; its automatic resume path validates the existing
checkpoint and continues the recorded trajectory. Do not use `--skip-wire` on
the first launch: the wiring check is part of the startup contract.

## Monitoring and evaluation

The trainer records an aggregated training journal every 4,194,304 tokens and
four-pass source validation every 8,388,608 tokens. It keeps two durable
checkpoint generations, saving at most every 16,777,216 tokens or 15 minutes.
The explicit milestone snapshots are:

| Snapshot | Purpose |
|---:|---|
| 100,007,936 | Compare the aggressive LR with the existing Stage-5 100M endpoint |
| 500,000,768 | First long-horizon capability check |
| 1,000,001,536 | Mid-run capability and forgetting check |
| 2,000,001,024 | Late-run capability trajectory |
| 2,499,999,744 | Final endpoint |

At each milestone, evaluate the same validation artifact and run the complete
five-shot suite in `evaluation/suites/full.yaml`. Use both ordinary task scores
and paired answer log-likelihood margins. Also record source-stratified NLL,
pass-1 versus later-pass NLL, and parameter drift from the wiring checkpoint.

For a safetensors milestone snapshot, the full-suite command is:

```bash
uv run python scripts/evaluate_lm_harness.py \
  --config benchmarks/core/stage_6_long_continuation/dense_swa_memory_attention_plastic_2p5b.yaml \
  --checkpoint <milestone>.safetensors \
  --suite evaluation/suites/full.yaml \
  --device cuda \
  --output <milestone>-full.json
```

The fixed Stage-5 references are:

- Dense SWA Memory Attention 100M:
  `benchmarks/core/stage_5_cloud_100m/results/bank_dense_100m/snapshots/model_000100007936.safetensors`
- SWA Transformer 100M:
  `benchmarks/core/stage_5_cloud_100m/results/vanilla_100m/snapshots/model_000100007936.safetensors`

The 100M comparison answers whether the aggressive LR changes the short-run
trajectory. The later milestones answer the substantive question: whether a
long fresh-data continuation produces capability gains, a plateau, or
distribution adaptation accompanied by forgetting.

If a later conservative branch is required, it must initialize from the same
5M wiring checkpoint and use the same data artifact, milestones, and schedule,
changing only the backbone peak LR to `3e-6`.
