# Stage 2: local Phase-B smoke

Stage 1 must be complete. Run:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_2_local_smoke
```

These 1M-token runs are integration gates, not final comparisons. Dense,
periodic-C32, and write-only explicit-`<MEM>`-C32 Bank each have their own arm.
Adaptive Recirculation and the Recirculation–Bank hybrid also have separate
arms. All local arms keep only the newest checkpoint generation.
