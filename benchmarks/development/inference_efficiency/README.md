# Inference-efficiency qualification

This benchmark measures the cost that determines when to run feedback
evaluation. It is separate from the scientific result: inputs are deterministic
ordinary vocabulary IDs, so no held-out data are required and no claims about
NLL or generation quality are made here.

The benchmark ran on the existing Verda spot VM `1A6000.10V` (NVIDIA RTX
A6000, 48 GB) with PyTorch 2.13.0+cu130. It timed:

1. Full-sequence K=1 and K=4 validation on the 2,048-token training block.
2. Cached standard K=1, cached feedback K=4, and exact cached K=4 decoding.
3. The exact-vs-feedback diagnostic at measurement time, including three per-token losses,
   hidden drift, cosine similarity, and the evaluator's host transfers.

The model-side timing curves were measured through 128 continuation tokens and
extrapolated linearly to the 2,047-token no-context continuation. The full
diagnostic uses the same extrapolation because its per-token host transfers make
a complete sweep unnecessarily expensive. Raw measurements are in:

- [FP32 Verda result](results/inference_efficiency_verda_a6000.json)
- [BF16 Verda result](results/inference_efficiency_verda_a6000_bf16.json)
- [Benchmark script](../../../scripts/benchmark_inference_efficiency.py)

## Measured FP32 costs

Across the four frozen-wiring variants:

| Workload | Per 2,048-token block | Existing 64-block check | Approx. 2M-token split |
| --- | ---: | ---: | ---: |
| Full K=4 validation | 0.255–0.279 s | 16.3–17.9 s | 4.2–4.5 min |
| Feedback-only cached continuation | 40–48 s | 42–51 min | 10.8–13.3 h |
| Full exact-vs-feedback diagnostic | 238–276 s | 4.2–4.9 h | 64.6–75.0 h |

The feedback variants have almost identical full-sequence validation costs;
the large gap is caused by one-token-at-a-time cached evaluation and, for the
diagnostic, repeated host-synchronizing metric work. BF16 reduced the 64-block
K=4 check to about 9–11 seconds, but made feedback-only and diagnostic timing
slightly worse on this A6000.

## Original diagnostic guidance

The following guidance concerns the combined exact-vs-feedback diagnostic, not
the subsequently implemented BOS-only NLL evaluator.

- Keep the existing routine K=4 validation cadence of 3,276,800 training
  tokens / 64 blocks. Its 16–18 second FP32 cost is operationally reasonable.
- Do not run the full feedback diagnostic in the training loop or against the
  full 2M-token split. At the current implementation cost, that would take
  days per arm.
- Run feedback evaluation only after a durable snapshot/checkpoint. Start with
  one full 2,048-token block (`prompt_tokens=1`, `continuation_tokens=2047`)
  per arm at an early snapshot and at the first milestone where the ordinary
  learning curve suggests a change. One block is roughly 4–5 minutes per arm.
- If a one-block result shows a meaningful separation, expand to 4–8 blocks at
  selected later snapshots. If it does not, defer more feedback evaluation
  until the evaluator is optimized or a more informative trained checkpoint is
  available.
- Reserve a complete feedback split for a final report or for after the
  per-token `.cpu()` metric transfers are removed and this qualification is
  rerun.

The 1-block result is a diagnostic, not a statistically stable downstream
claim. The purpose of this schedule is to detect whether feedback behavior is
emerging without allowing evaluation cost to dominate the 100M-token training
trajectories.

## Scope and implementation status

The four measured variants include the older middle-layer `recirculation`, not
the two new late-memory recurrent mergers. The continuation totals are
extrapolations from a 128-token horizon, not direct full-block measurements.
Do not transfer these timings to the new mergers without qualification.

The existing packed-data diagnostic takes its prefix from the block:
`prompt_tokens=1` means its first data token, not an inserted BOS. A true
BOS-only evaluator now consumes ordinary text with the same feedback inference
implementation and reports full and aligned target coverage. The 4–5 minute
estimate includes exact/reference decoding and
diagnostic metrics, whereas feedback-only cost is reported separately above.

Paper readout/replay execution is removed. Cleanup 3–4 fixes state/naming and
shares evaluation precision/provenance; the historical timing files above are
unchanged. New snapshot publication is interruption-safe.
Optional selected-checkpoint feedback now pauses training after a durable
snapshot, outside measured optimizer time, initially one fixed full block per
arm. It does not run at every routine validation. This implementation has not
been re-timed on the A6000. See
[the cleanup tracker](../../../docs/CLEANUP_STATUS.md) for remaining additions.

The current script defaults to the five active arms and emits schema-2 rows and
cost summaries keyed by arm, so the two `recurrent_memory` mergers remain
distinct. The retained JSON files predate this change; they have not been
rewritten or relabelled. Timing the actual BOS-only NLL evaluator is the next
qualification task, separate from the synthetic cached-continuation curves.
