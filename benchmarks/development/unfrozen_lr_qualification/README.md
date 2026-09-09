# Unfrozen learning-rate qualification

This development study qualifies one shared backbone learning rate for the
unfrozen long-run comparison. It uses six vanilla arms over `3e-6`, `1e-5`,
`3e-5`, `1e-4`, `3e-4`, and `1e-3`. The selected rate is then carried into
recurrent and Memory Attention runs; those models keep their added-parameter
learning rate at `1e-3`.

Each arm processes 20,054,016 linguistic tokens with the same effective batch
of 65,536 tokens. The qualification is intentionally vanilla-only: it measures
the backbone optimizer without letting architecture-specific feedback gradients
choose a different pretrained-backbone rate. Architecture-specific stability
remains a diagnostic during the later comparison runs.

The short schedule warms to the candidate backbone rate over 5,013,504 tokens
and then holds it. Selection uses validation NLL, finite loss/gradient
behavior, and trajectory stability. The `1e-4`–`1e-3`
arms are deliberately stress tests for unfrozen-backbone instability.

No BOS-feedback generation runs during qualification. Before repeating the study,
materialize and pin `data/dolmino/unfrozen_lr_20m_2048`, change the study to `locked`, and
run the normal CUDA preflight. Qualification archives metrics, journals, and
configuration metadata only; rolling checkpoints stay on the VM long enough to
support interruption recovery and are not copied into the performance results.
After recording the winner, copy that rate into both `learning_rate` and
`pretrained_learning_rate` for every selected `unfrozen_scaling_core` arm, set
`learning_rates_qualified: true`, and lock that study.

## Completed qualification

All six arms completed 20,054,016 token presentations and 306 optimizer updates.
The provisional shared backbone rate is `3e-4`; it had the lowest NLL at all
four recorded validation points. Each arm improved monotonically, and every
logged training loss and gradient norm was finite.

| Backbone LR | Final validation NLL |
| --- | ---: |
| `3e-6` | 3.294322 |
| `1e-5` | 3.158117 |
| `3e-5` | 3.054958 |
| `1e-4` | 2.977201 |
| **`3e-4`** | **2.963240** |
| `1e-3` | 3.077516 |

These are teacher-forced K=1 scores with BF16 autocast on the first 64
validation blocks: 131,008 scored tokens from the artifact pinned in
`STUDY.yaml`. Final scores are also each arm's best recorded scores. The
`3e-4` advantage over `1e-4` is 0.013961 nats per token. This single-seed,
short-budget qualification supports a provisional shared rate; it does not
qualify the long-run cosine schedule or establish an optimum for feedback models.

The table summarizes the local `results/<arm>/metrics.jsonl` and
`segments.jsonl` records; raw telemetry and weights remain ignored. The `1e-5`
arm resumed after a signal and completed, and the `3e-6` arm additionally saved
durable snapshots. All runs record execution-source SHA-256
`d45759489b29288aa87db0ea7f5ce7e3b076b714de4485b0a93bb204758ad7a0`
and lockfile SHA-256
`4334c465f3b39032ea8378b556f4a030ec6f5d30bb9d2b8917a75995c9e03fef`.
Their Git revision metadata is unavailable, so these results must not be
attributed to the later runtime-cleanup commits.
