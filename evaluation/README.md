# Evaluation suites

Reusable `lm-evaluation-harness` task suites live here. They are not tied to a
single benchmark study, so the suite definitions are kept separate from both
training configs and data artifacts.

- `suites/quick.yaml`: small development sanity battery.
- `suites/full.yaml`: candidate-ranking battery aligned with the tasks used in
  the Full-Bandwidth Transformer comparison. It scores supplied candidates by
  conditional log-likelihood; it is not free generation.
- `suites/generation_math.yaml`: long free-generation math evaluation.
- `suites/generation_code.yaml`: free code generation whose metrics execute
  generated programs; use only in an isolated environment.

Evaluation output should be written beside the benchmark/checkpoint being
studied when it is worth retaining; otherwise use a temporary path.

`scripts/evaluate_lm_harness.py` requires both `--prefill-passes K` and
`--decode-mode standard|feedback`. K applies only to prompt construction. In
feedback mode, each observed candidate token or newly generated token advances
the feedback state once. Standard mode is the decode ablation. Vanilla accepts
only K=1 standard mode; unsupported feedback requests fail instead of silently
falling back.

The evaluator requires a checkpoint by default. Use `--initialized-baseline`
only for a deliberately labelled time-zero result. Retained JSON includes
checkpoint/config/source/suite hashes, package versions, seeds, task config,
raw sample records, and extractable candidate margins.
For configs with `init_from`, time zero includes those wired weights but starts
before the new optimizer trajectory, matching the trainer's initialization
semantics.

Validation NLL is a separate lane. `scripts/evaluate_pass_depth.py` performs
exact full-sequence teacher forcing for K=1 through K=8 by default, with no
generation or collapsed continuation. Supply `--evaluation-data-dir` for the
independent artifact used for final claims.

The current harness path covers ordinary-token variants. Explicit memory-token
Memory Attention evaluation still requires a physical `<MEM>` insertion schedule; the
ordinary text-task adapter does not silently insert control positions.
