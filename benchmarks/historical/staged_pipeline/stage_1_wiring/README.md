# Stage 1: local frozen-backbone wiring

Run all canonical wiring arms sequentially:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_1_wiring
```

Prepare and verify `data/dolmino/wiring_2048` first. Every arm consumes its
5,242,880-token training split exactly once with an added-parameter learning
rate of `1e-4`.

Use `--arm <id>` to run one arm. Generated checkpoints and metrics live below
`results/<arm>/` and are ignored by Git. Memory Attention is wired separately for
dense, strided-C32, and write-only explicit-`<MEM>`-C32 writes. The strided
hybrid is wired with adaptive Recirculation. Multiscale Memory Attention has a separate
wiring arm. Strided Attention has no added parameters and does not use Phase A.

These local configs keep only the newest checkpoint generation. Copy completed
wiring checkpoints to durable storage before removing local run directories.
The `_5m` arm IDs create distinct output directories and prevent accidental
resume from the superseded runs that used the old 1M artifact.
