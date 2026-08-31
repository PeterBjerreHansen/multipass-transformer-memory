# FBT adaptation program

This ad-hoc study compares the repository's original FBT fusion with the
stabilized training recipe in the [Full-bandwidth transformer
paper](https://arxiv.org/abs/2608.08888). The paper-style arms normalize the
token embedding before the gate, RMS-normalize the fused input, apply
training-only `Uniform[-0.02, 0.02]` carried-state jitter, and use a random plain
prefix on every feedback pass.

## 1. Untouched fusion diagnostic

Run the two fusion modes with identical base and feedback weights on the same 32
validation blocks:

```bash
uv run python benchmarks/ad_hoc/fbt/diagnose_fusion.py \
  --config benchmarks/ad_hoc/fbt/current_phase_b_lr3e7.yaml \
  --max-blocks 32 \
  --output benchmarks/ad_hoc/fbt/results/fusion_baseline.json
```

The report includes embedding/fused RMS, gate-logit standard deviation, NLL at
passes 1-4, and hidden-state deltas. Jitter is a training regularizer, so the
evaluation comparison isolates gate-input normalization while recording the two
training jitter settings.

## 2. Primary Phase-B screen

The primary screen compares the current fusion control and paper-style backbone
learning rates `1e-7`, `3e-7`, and `1e-6`. Every Phase-B arm uses added-parameter
LR `1e-4`, the paper's `75% K=1 / 22% K=2 / 3% K=3` mixture, and pass weights
`[1]`, `[0.5, 0.5]`, and `[0.5, 0.25, 0.25]`.

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/ad_hoc/fbt \
  --arm current_phase_b_lr3e7 \
  --arm paper_phase_b_lr1e7 \
  --arm paper_phase_b_lr3e7 \
  --arm paper_phase_b_lr1e6
```

`run_study.py` uses exact automatic resume. Use `--until-unique-tokens 262144`
for a single evaluation interval when manually reviewing each increment; later
targets must remain multiples of 262,144.

## 3. Controls and escalation

Run the frozen-backbone Phase-A control separately:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/ad_hoc/fbt \
  --arm paper_phase_a_control
```

Phase A omits K=1 batches because the added FBT path is absent on pass 1. Its
K=2/K=3 probabilities preserve the paper mixture's `22:3` ratio. If the best
Phase-B arm adapts too slowly, run the predeclared feedback-heavy arm, which
doubles feedback-bearing batch mass while preserving that ratio:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/ad_hoc/fbt \
  --arm paper_phase_b_heavy_lr3e7
```

## Validation gates

All arms evaluate four passes on the same 32 validation blocks every 262,144
unique tokens. Training stops after a checkpoint when all gates pass:

- pass-4 NLL is at most `2.34`;
- pass-4 NLL is no more than `0.02` above pass-2 NLL;
- hidden-state delta RMS is nonincreasing from pass to pass; and
- pass-1 NLL is at most `2.573` (the measured `2.553` base plus `0.02`).

Each validation record in `metrics.jsonl` contains the actual values and status
of every gate. Generated runs remain ignored; this study's configs, diagnostic,
manifest, and results README are explicitly visible to Git.
