# NMP continuation from the 100M hybrid

This is the first committed matched-control NMP study. Every arm starts from
the exact completed adaptive-Recirculation + periodic-Memory-Attention
checkpoint at 100,007,936 tokens and continues on the same 90% K=2 / 10% K=3
pass mixture used to train the parent.

The primary comparison is deliberately narrow:

- `ntp_control`: ordinary NTP continuation with no predictor.
- `bank_nmp_coupled`: shared-final periodic Memory-Attention NMP with gradients
  through the predictor input into the model.
- `bank_nmp_head_only`: the same target, coefficient, head, and forward work,
  but the predictor input is detached so NMP cannot regularize the model.

`shared_final` is the default objective. For a sampled K, every active NMP
predictor is supervised by the final pass's future memory target. NTP and NMP
use the parent's pass weighting: `[0.1, 0.9]` at K=2 and `[0.1, 0.0, 0.9]` at
K=3. Fixed validation evaluates NMP separately at K=2 and K=3.

The study ceiling is 52,428,800 new tokens per arm. The 10,485,760- and
26,214,400-token snapshots are trajectory checkpoints, not small-run
substitutes. The primary comparison is the common 52,428,800-token endpoint.

## Required preflight

1. Materialize and verify `data/dolmino/nmp_100m_2048`. It contains 52,428,800
   training tokens after skipping the 5,242,880-token wiring slice and the
   completed 100,007,936-token Phase-B slice.
2. Run `make check` and the study verifier.
3. Run `calibrate_nmp.py` on CUDA after isolated head warm-up. It measures K=2,
   K=3, and the exact configured 90/10 gradient mixture. The committed
   coefficient `0.8510068634` is only the legacy 10M-parent 5% shared-gradient
   coefficient. Replace it in both NMP configs if the 100M mixed-pass
   calibration differs materially, and commit the compact report.
4. Wire all three arms at both sampled pass depths.

The continuation preserves the parent's terminal mature-module learning rates:
`1e-7` for pretrained parameters and `3e-6` for existing Memory Attention and
Recirculation parameters. The new predictor has its own `3e-5` optimizer group.
All groups use a constant schedule over the 50 Mi continuation.

```bash
uv run python scripts/prepare_data.py \
  --config data/dolmino/nmp_100m_2048/config.yaml
uv run python scripts/verify_data.py data/dolmino/nmp_100m_2048
make check
uv run python benchmarks/development/nmp_100m_continuation/calibrate_nmp.py \
  --config benchmarks/development/nmp_100m_continuation/bank_nmp_coupled.yaml \
  --output benchmarks/development/nmp_100m_continuation/results/calibration.json
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/nmp_100m_continuation \
  --wire-only
```

## Execution

Run the complete matched triplet. Stop only for invalid execution, such as
non-finite loss, unrecoverable resource failure, corrupt artifacts, or a
predeclared catastrophic-loss threshold. Do not select arms using the 10M
snapshot and do not compare checkpoints with different token counts.

```bash
# Operational checkpoint: all three arms to 10 Mi tokens.
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/nmp_100m_continuation \
  --skip-wire --until-unique-tokens 10485760

# Interim checkpoint: all three arms to 25 Mi tokens.
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/nmp_100m_continuation \
  --skip-wire --until-unique-tokens 26214400

# Primary endpoint: all three arms to 50 Mi tokens.
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/nmp_100m_continuation \
  --skip-wire --until-unique-tokens 52428800
```

At common-token checkpoints, compare validation NLL at fixed K=1 through K=4,
held-out NMP at fixed K=2 and K=3, event-balanced and query-weighted NMP
diagnostics, and target drift. Lower online NMP loss alone is not evidence of a
better language model. The coupled arm must separate from both ordinary NTP
continuation and the detached-input placebo.

See `HANDOFF_PLAN.md` for the implementation and execution handoff.
