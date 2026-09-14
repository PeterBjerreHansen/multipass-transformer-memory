# Unfrozen benchmark decision record

Updated 2026-09-11. This note records the scientific and implementation choices
behind the current unfrozen development studies and their staged execution.

## Scientific question

The long study asks whether the frozen-wiring result survives when the whole
model adapts, and whether dense content-addressed feedback offers a favorable
data/parameter/compute tradeoff against a reasonable recurrent merger and
continued vanilla training. It is a conceptual comparison, not a frontier or
SOTA claim. Because every backbone parameter trains, it also does not support a
low-optimized-parameter claim.

## Core arms

The principal study contains only three fresh Phase-B trajectories:

1. vanilla K=1 continued pretraining;
2. the current two-site adaptive recirculation merger, with controller width
   660; and
3. dense Memory Attention with two reader sites `[3, 7]`, a 32-record window,
   16 KV heads, `aligned_gqa` initialization, destination-gated fusion, and
   controller width 64.

The private fusion pilot favored the aligned destination-gated reader at the
chosen development budget. This evidence selects a representative; it is not a
claim that the fusion or initialization is globally optimal. No-memory,
projected-residual, strided, dense-plus-strided, and external-paper controls are
optional studies outside the core.

The feedback mechanisms are reasonably parameter matched without dummy
parameters. Instantiation gives:

| Arm | Added parameters | Total parameters |
| --- | ---: | ---: |
| Vanilla | 0 | 248,024,064 |
| Adaptive recirculation | 7,341,424 | 255,365,488 |
| Dense Memory Attention | 7,756,032 | 255,780,096 |

The feedback added-parameter ratio is 1.0565 and the total-model difference is
about 0.16%. Sixteen KV heads preserve a full-width, ordinary multi-head reader;
it was not narrowed merely to force an exact count.

## Training and compute contract

All arms start from the same pretrained checkpoint. The backbone and additions
train from token zero; there is no frozen prefix. Feedback arms sample K=2 with
probability 0.9 and K=3 with probability 0.1 per microbatch, and only the final
pass receives NTP loss. Vanilla uses K=1. The nominal optimizer batch is 65,536
linguistic tokens.

Feedback arms consume exactly 2,500,001,792 token presentations from one fixed
artifact. Vanilla uses the same artifact, then reshuffles and repeats it until
5,350,227,968 presentations. It receives more presentations but no additional
corpus. Under the current dominant-matmul estimator, recurrent and dense
feedback cost 2.13731x and 2.14009x vanilla K=1 respectively.

One vanilla trajectory supplies both data- and compute-axis comparisons. The
five named stages use the nearest attainable 65,536-token optimizer-batch match
to dense Memory Attention:

| Feedback target | Vanilla target |
| ---: | ---: |
| 100,007,936 | 214,040,576 |
| 500,039,680 | 1,070,137,344 |
| 1,000,013,824 | 2,140,143,616 |
| 2,000,027,648 | 4,280,221,696 |
| 2,500,001,792 | 5,350,227,968 |

The vanilla config also retains the closest recurrent-compute snapshots and the
equal-data 2.5B point. The generated `stage_budget.json` records the exact
estimator inputs and residual rounding error. Named-stage compute coverage is
within 0.008% of the dense target.

Cosine schedules use each arm's final presentation horizon, which aligns
schedule progress at compute-matched stages. It does not align schedule progress
at the vanilla 2.5B equal-presentation snapshot. That point remains useful, but
its schedule-position difference must accompany any data-axis interpretation.

## LR qualification

The old Phase-B sweep and frozen `1e-3` result do not qualify backbone rates for
this experiment. A separate 20,054,016-token study sweeps six backbone rates
(`3e-6`, `1e-5`, `3e-5`, `1e-4`, `3e-4`, and `1e-3`) in the vanilla family.
The selected shared backbone rate is then applied to the recurrent and Memory
Attention core arms; their added-parameter rate remains `1e-3`. This is an
equal-budget magnitude qualification, not a claim that the vanilla ranking is
architecture-independent.

The qualification schedule warms for 5,013,504 tokens and then holds the peak
rate. Selection uses validation NLL plus finite-loss, gradient, and trajectory
stability checks. It omits expensive BOS-feedback decoding. A lower added-rate
rescue is allowed only as a new declared run if `1e-3` is unstable.

The qualification selected the shared backbone rate of `3e-4`, which is recorded
in the long configs. The long artifact is materialized and pinned, the
target-GPU checks passed, and the locked study manifest sets
`learning_rates_qualified: true`. The staged cloud run is therefore active;
stage boundaries remain operational checkpoints, not separate scientific
experiments.

## Data and evaluation

Three new recipes separate LR selection, long training, and final evidence:

- `unfrozen_lr_20m_2048`: 20M training plus 2M monitoring tokens;
- `unfrozen_2p5b_2048`: the shared 2.5B corpus plus 2M monitoring tokens; and
- `unfrozen_final_eval_2048`: 10M validation tokens after a 3B stream offset.

Materialized manifests must be pinned before execution. Complete-document
disjointness must be checked after materialization; the source offset is useful
separation, not proof of disjointness.

Routine validation reports K=1 for vanilla and parallel K=1 through K=4 for
feedback models. Selected 100M, 1B, and final feedback snapshots additionally
run the production feedback evaluator. Final paper evaluation uses the separate
held-out artifact. Parallel K=4 and Live Feedback remain distinct estimands: a
K=4 gain without stable Live Feedback is a failure of the deployment surrogate,
not a positive temporal-memory result.

## Staging and recovery

`STUDY.yaml` now supports named per-arm targets. Validation requires targets to
be increasing, attainable whole-microbatch boundaries, durable snapshot
thresholds, and final config horizons. `run_study.py --stage` and
`run-cloud-study --stage` translate them to the trainer's exact stop boundary.

The final horizon and cosine schedule are fixed from the first stage. Resume
restores model, optimizer, sampler, pass schedule, RNG, and counters without
changing the schedule. Intermediate cloud stages retain remote checkpoints and
require target-aware local transfers; only a final stage deletes verified
remote output.

## Launch gates

Before LR qualification:

1. materialize and verify the 20M artifact;
2. pin its manifest hash and lock the qualification study;
3. run target-CUDA Phase-B checks with real K=3 updates, optimizer allocation,
   K=4 validation, snapshot creation, interruption, and resume.

Before scaling (now satisfied for the active study):

1. record the single shared LR winner in the long configs;
2. materialize, verify, pin, and disjointness-check the long and final artifacts;
3. regenerate and review `stage_budget.json`;
4. perform the same target-CUDA preflight on all core arms; and
5. set `learning_rates_qualified: true`, promote to a locked core study, and
   review the clean source state.

Qualification alone did not authorize launch; the additional data, preflight,
and source-state gates above were completed before the current staged run.
