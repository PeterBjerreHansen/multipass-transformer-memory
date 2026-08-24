# Evaluation suites

Reusable `lm-evaluation-harness` task suites live here. They are not tied to a
single benchmark study, so the suite definitions are kept separate from both
training configs and data artifacts.

- `suites/quick.yaml`: small development sanity battery.
- `suites/full.yaml`: broader base-model battery aligned with the tasks used in
  the Full-Bandwidth Transformer comparison.

Evaluation output should be written beside the benchmark/checkpoint being
studied when it is worth retaining; otherwise use a temporary path.

For multipass checkpoints, `scripts/evaluate_lm_harness.py` uses recurrent
inference by default with a two-pass prompt prefill. The prompt is refined K
times, then the continuation is scored or generated one token at a time from a
single collapsed recurrent cache. Use `--prefill-passes K` to change K, or
`--inference-mode forward` to measure the ordinary public one-pass path.

The current harness path covers ordinary-token variants. Explicit memory-token
Memory Attention evaluation still requires a physical `<MEM>` insertion schedule; the
ordinary text-task adapter does not silently insert control positions.
