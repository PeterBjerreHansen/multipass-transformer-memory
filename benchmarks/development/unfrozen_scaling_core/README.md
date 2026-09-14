# Unfrozen scaling core

This is the staged three-arm scaling study:

1. a vanilla K=1 backbone trained for a compute-matched token budget;
2. the two-site adaptive recirculation merger selected for the recurrent class;
3. two-site dense Memory Attention with 32 records, 16 KV heads,
   `aligned_gqa` initialization, and destination-gated fusion.

The two feedback arms see exactly 2,500,001,792 linguistic training-token
presentations. The vanilla arm cycles the same 2.5B-token artifact and continues
to 5,350,227,968 presentations. That endpoint covers the slightly more expensive
Memory Attention arm under the repository's dominant-matmul FLOP estimator.
The recurrent arm has its own exact compute-match snapshots in the vanilla
config; the named stages conservatively use the dense-attention targets.

Both feedback arms use `[3, 7]` injection sites and the same K distribution
(`K=2` with probability 0.9, `K=3` with probability 0.1). Only the final pass
receives NTP loss. Added parameters use `1e-3`; the six-arm vanilla qualification
selected the shared backbone rate of `3e-4`. The long-study artifact is pinned
and the study is locked with `learning_rates_qualified: true`, so the staged
run may resume from its declared boundaries. This qualification does not make
the shared rate architecture-optimal; the final report must retain its scope.
Cosine decay is normalized to each arm's final horizon. Thus the named
compute-matched stages are also approximately schedule-progress matched. The
vanilla equal-presentation snapshot at 2.5B occurs earlier in its schedule and
must be reported with that qualification.

## Stages

| Stage | Feedback tokens | Vanilla tokens |
| --- | ---: | ---: |
| `early_100m` | 100,007,936 | 214,040,576 |
| `pilot_500m` | 500,039,680 | 1,070,137,344 |
| `one_billion` | 1,000,013,824 | 2,140,143,616 |
| `two_billion` | 2,000,027,648 | 4,280,221,696 |
| `final_2p5b` | 2,500,001,792 | 5,350,227,968 |

All final horizons are configured from the first launch. `--stage` only stops a
run at a declared snapshot, so the cosine schedule and optimizer state resume
unchanged. For example, after data hashes and LRs are locked:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/unfrozen_scaling_core \
  --stage early_100m
```

On cloud infrastructure, pass the same `--stage` to `scripts/run-cloud-study`.
Intermediate cloud stages retain the VM and remote output so the next stage can resume;
the final stage removes it only after verified transfer.

Routine validation reports K=1 for vanilla and K=1 through K=4 behavior for the
feedback models. Expensive feedback decoding is limited to the 100M, 1B, and
final feedback snapshots. Final paper evaluation must use the independent
`data/dolmino/unfrozen_final_eval_2048` artifact and report curves against both
linguistic token presentations and estimated training FLOPs.

Plots for this study are ephemeral reports, not reusable result files. Every
plot request must read the current `results/<arm>/metrics.jsonl` files first,
record the study and arm paths in the figure metadata or caption, and be
regenerated immediately before display. Do not reuse PNG/HTML files from the
Codex visualization cache or from retired study directories.

The held-out evaluation manifest SHA-256 is
`79d38771a3803f173a762bfbfae969f6a382bc6cea6b9036d4e4fbe03f119a2e`.
It is pinned here because it is consumed by separate final evaluation commands;
`STUDY.yaml` pins the training artifacts consumed by its arms.

Promotion requires: materialized and verified data, pinned manifest hashes,
recorded LR winners, `learning_rates_qualified: true`, CUDA preflight, generated
parameter/FLOP reports, and a locked manifest under `benchmarks/core`.
