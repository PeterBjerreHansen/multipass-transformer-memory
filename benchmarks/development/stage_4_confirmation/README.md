# Stage 4: selected confirmation

First resume each promoted Stage-3 seed to its declared 10M endpoint:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_3_cloud_pilot \
  --skip-wire --arm vanilla_seed1337 --arm bank_periodic32_seed1337 \
  --arm recirculation_adaptive_seed1337
```

The command shows the active periodic Memory Attention and adaptive Recirculation arms as
examples; replace the selected arm IDs with the Stage-3 gate decisions.

Then materialize the two additional-seed study. Choose the fast baseline and
hybrid using the Stage-3 gate:

```bash
uv run python benchmarks/development/stage_4_confirmation/prepare.py \
  --fast recirculation_adaptive --memory-attention periodic32 --hybrid recirculation
uv run python scripts/verify_study.py \
  benchmarks/development/stage_4_confirmation
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_4_confirmation --skip-wire
```

`--fast` accepts only `recirculation_adaptive`; `--memory-attention` (historical
alias `--bank`) accepts `dense`,
`periodic32`, or `memory_token32`; `--hybrid` accepts `recirculation` and
defaults to `recirculation`. Preparation fails rather than overwriting an
existing selection; pass `--force` only when deliberately replacing a study
that has not begun execution.
