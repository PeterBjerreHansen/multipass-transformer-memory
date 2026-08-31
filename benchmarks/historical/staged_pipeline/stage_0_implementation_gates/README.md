# Stage 0: implementation gates

Run from the repository root:

```bash
uv run pytest -q
uv run python scripts/verify_study.py
git diff --check
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_1_wiring \
  --wire-only --wire-device mps
```

The final command constructs every architecture and runs both sampled pass
depths through one forward/backward preflight. Freeze a clean source commit
before starting Stage 1.
