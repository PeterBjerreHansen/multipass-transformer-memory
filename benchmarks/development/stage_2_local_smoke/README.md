# Stage 2: local Phase-B smoke

Stage 1 must be complete. Run:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_2_local_smoke
```

These 1M-token runs are integration gates, not final comparisons. Dense,
periodic-C32, and write-only explicit-`<MEM>`-C32 Memory Attention each have
their own arm. Adaptive Recirculation and the Recirculation–Memory Attention
hybrid also have separate arms. Multiscale Memory Attention initializes from
its Stage-1 wiring checkpoint. Sparse
SWA starts directly in Phase B because it adds no parameters. All local arms
keep only the newest checkpoint generation.
