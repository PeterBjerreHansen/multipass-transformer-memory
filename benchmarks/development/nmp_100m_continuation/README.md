# NMP continuation from the 100M hybrid

This is the first committed matched-control NMP study. It starts every arm
from the exact completed adaptive-Recirculation + periodic-Memory-Attention
checkpoint at 100,007,936 tokens and trains at fixed K=2 on the next fresh 5M
Dolmino tokens.

The primary comparison is deliberately narrow:

- `ntp_control`: ordinary NTP continuation with no predictor.
- `bank_nmp_coupled`: shared-final periodic Memory-Attention NMP with gradients
  through the predictor input into the model.
- `bank_nmp_head_only`: the same target, coefficient, head, and forward work,
  but the predictor input is detached so NMP cannot regularize the model.

The study does not enable recurrent NMP. Fixed K=2 removes sampled teacher-depth
variation; `shared_final` still includes explicit pass-1 to pass-2
self-distillation and is labeled accordingly. A later study may compare
`same_pass` and final-pass-only objectives if the coupled arm beats both
controls.

## Required preflight

1. Materialize and verify `data/dolmino/nmp_100m_2048`. Its skip count is the
   5,242,880-token wiring slice plus the completed 100,007,936-token Phase-B
   slice, so it does not restart the source checkpoint's training stream.
2. Run `make check` and the study verifier.
3. Run `calibrate_nmp.py` on CUDA. The committed coefficient `0.8510068634` is
   the conservative 5% shared-gradient coefficient from the legacy 10M parent,
   not a claim about the 100M gradient scale. Replace it in both NMP configs if
   the post-head-warm-up 100M calibration differs materially, and record the
   resulting report before execution.
4. Wire all three arms before paid training.

```bash
uv run python scripts/prepare_data.py \
  --config data/dolmino/nmp_100m_2048/config.yaml
uv run python scripts/verify_data.py data/dolmino/nmp_100m_2048
uv run python benchmarks/development/nmp_100m_continuation/calibrate_nmp.py \
  --config benchmarks/development/nmp_100m_continuation/bank_nmp_coupled.yaml \
  --output benchmarks/development/nmp_100m_continuation/results/calibration.json
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/nmp_100m_continuation \
  --wire-only
```

## Execution and decisions

Run every arm to the common 1M gate. Compare checkpoints at the same token
counts; reaching the parent loss is not a stopping criterion. Stop an arm early
only for non-finite behavior or a predeclared NTP-harm gate.

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/nmp_100m_continuation \
  --skip-wire --until-unique-tokens 1048576
```

Promote all three arms together to 5M only if the coupled arm remains
competitive in held-out NTP and improves held-out NMP beyond the head-only
placebo. Fixed-K validation reports both query-weighted and event-balanced NMP
statistics. Any small NTP advantage requires multiple seeds before it is
interpreted.

Use `evaluate_target_drift.py` on common-token checkpoints to determine whether
lower online NMP loss reflects improved prediction, an easier moving target, or
both.
