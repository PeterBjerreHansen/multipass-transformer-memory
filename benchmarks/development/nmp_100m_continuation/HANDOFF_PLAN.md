# Coding-agent handoff: 100M NMP continuation

## Goal

Finish and execute a matched NMP continuation study from the exact 100M hybrid
checkpoint. The primary NMP objective is `shared_final`. Training samples K=2
with probability 0.9 and K=3 with probability 0.1. The primary endpoint is 50
Mi new tokens per arm; 10 Mi and 25 Mi are common trajectory checkpoints.

Do not reintroduce the retired ad-hoc NMP campaign. Preserve unrelated FBT work
and any other pre-existing worktree changes.

## Locked scientific contract

Use these three arms:

1. `ntp_control`: no NMP head or NMP loss.
2. `bank_nmp_coupled`: Bank NMP with `nmp_detach_predictor_input: false`.
3. `bank_nmp_head_only`: identical Bank NMP configuration with
   `nmp_detach_predictor_input: true`.

Keep the following identical across arms: source checkpoint, continuation data,
training seed, architecture seed, batch size, pass-scheduler seed, optimizer
restart, NTP pass weights, learning-rate schedule, checkpoint cadence, and token
budget. Only the NMP fields declared in `STUDY.yaml` may differ.

The target mode must remain:

```yaml
nmp_target_mode: shared_final
```

The pass contract must remain:

```yaml
pass_schedule:
  - probabilities: {2: 0.9, 3: 0.1}
ntp_pass_loss_weights_by_k:
  2: [0.1, 0.9]
  3: [0.1, 0.0, 0.9]
bank_nmp_pass_loss_weights_by_k:
  2: [0.1, 0.9]
  3: [0.1, 0.0, 0.9]
```

Do not add `same_pass`, recurrent NMP, dual NMP, an EMA/frozen teacher, or a
coefficient sweep to this first campaign.

## Implementation work before training

### 1. Verify the checked-in study edits

- Confirm all three configs have `max_unique_tokens: 52428800`.
- Confirm snapshots are exactly 10,485,760, 26,214,400, and 52,428,800.
- Confirm fixed NMP validation runs at K=2 and K=3.
- Confirm `data/dolmino/nmp_100m_2048/config.yaml` materializes 52,428,800 fresh
  training tokens with `train_skip_tokens: 105250816`.
- Run the study verifier and ensure the only triplet differences are the NMP
  fields declared in `STUDY.yaml`.

### 2. Run mixed-pass calibration

`calibrate_nmp.py` uses the normalized pass probabilities from the config. Its
report includes:

- separate K=2 and K=3 measurements;
- the probability-weighted aggregate;
- NTP and NMP gradient norms for pretrained, existing memory/recirculation, and
  predictor-head parameter groups;
- NTP/NMP gradient cosine for the shared groups;
- candidate coefficients for 5%, 10%, and 20% NMP-to-NTP pretrained-gradient
  ratios after head-only warm-up.

The script measures the same fixed blocks at each K and differentiates the
exact probability-weighted loss. It does not estimate the 90/10 mixture from a
small random sample.

Use the post-warm-up 5% coefficient for this campaign unless it is non-finite
or clearly pathological. Write the chosen value into both NMP configs and
record the calibration JSON under the ignored results directory; commit a
compact reviewed summary, not large artifacts.

### 3. Verify the separate predictor LR

The current optimizer puts the newly initialized NMP head and all already
trained memory/recirculation parameters in one `added` group. A 50 Mi
continuation should not require resetting the mature memory modules to the
head's learning rate.

The optional `nmp_predictor_learning_rate` field creates a third optimizer
group named `nmp_predictor`. Verify these requirements remain true:

- default `null` preserves current behavior and old configs;
- when set, only recurrent/Bank NMP predictor parameters use this group;
- existing memory writers/readers and the adaptive recirculation controller
  remain in `added`;
- no parameter may appear in more than one group;
- optimizer checkpoint resume and group-metadata repair must support the new
  group;
- semantic `init_from` compatibility must treat the field as a trajectory
  setting, not an architecture setting;
- add the field to the study's declared NMP experimental axes if it differs
  between the control and NMP configs;
- tests must cover grouping, LR scheduling, serialization, and exact resume.

The locked values preserve the 100M checkpoint's terminal rates: `1e-7` for
pretrained parameters, `3e-6` for mature Memory Attention/Recirculation
parameters, and `3e-5` for the fresh predictor. The control has no predictor
group.

### 4. Verify the 50 Mi learning-rate schedule

The locked constant schedule preserves the terminal mature-module rates over
the full 52,428,800-token horizon while the NMP loss itself ramps over its
configured warm-up. Do not replace it with a short-run schedule accidentally.

The scheduler must remain anchored to the 50 Mi ceiling when a run is paused at
10 Mi or 25 Mi and resumed. Verify an interrupted/resumed run has the same LR
and pass-scheduler trajectory as an uninterrupted run.

### 5. Add a compact comparison report

Create a script that consumes common-token validation records from the three
arms and writes one machine-readable JSON plus a short Markdown table. At 10,
25, and 50 Mi, report:

- fixed-K NLL and perplexity for K=1, 2, 3, and 4;
- coupled-minus-control and coupled-minus-head-only NLL deltas;
- held-out Bank NMP loss at K=2 and K=3;
- event-balanced and query-weighted NMP metrics;
- target RMS/feature spread and valid event counts;
- target-drift cosine, CKA, RMS difference, and covariance-spectrum summaries;
- realized pass histogram and optimizer-group LRs.

Fail clearly if compared checkpoints have different unique-token counts,
source checkpoint hashes, validation data hashes, or execution-critical config
fields.

## Verification gates

Before paid training:

1. Run focused tests for config, optimizer grouping/resume, mixed-pass NMP,
   calibration, validation, and study manifests.
2. Run `make check`.
3. Materialize and verify the 50 Mi continuation data.
4. Wire every arm at K=2 and K=3.
5. Run a very short CUDA smoke execution only to validate memory use, logging,
   checkpoint recovery, and finite gradients. Do not interpret its loss.
6. Re-run the mixed-pass calibration and lock the coefficient and LR settings.

Record the source checkpoint SHA-256, data-manifest SHA-256, git commit, CUDA and
PyTorch versions, device type, calibrated coefficient, and final configs.

## Training and decisions

Run all three seed-1337 arms to the same 50 Mi endpoint. The 10 Mi checkpoint is
operational and the 25 Mi checkpoint is interim; neither replaces the endpoint.
Do not promote only a favorable arm. Stop early only for a predeclared invalid
run condition such as non-finite values, corrupted recovery, or catastrophic
and sustained NTP divergence.

The primary comparison is fixed-K validation NLL at 50 Mi. Also inspect the NLL
trajectory from 10 to 50 Mi so a single noisy endpoint does not dominate.
Interpretation:

- coupled better than both controls: evidence that NMP gradients into the model
  help; proceed to replication;
- coupled matches head-only: the predictor machinery or ordinary continuation
  explains the result; do not claim representation improvement;
- both NMP arms beat NTP while matching each other: investigate compute or
  initialization effects before attribution;
- coupled hurts NLL while NMP loss falls: the auxiliary task is being optimized
  but is misaligned with language modeling;
- NMP loss falls with large target drift: investigate moving-target adaptation
  before interpreting predictability.

If the seed-1337 triplet gives a meaningful coupled advantage, replicate the
entire triplet with at least two additional training/pass-scheduler seeds. Keep
the parent checkpoint and data artifact fixed, and match each seed across its
three arms. Run confirmations to the same 50 Mi endpoint. Do not begin the
`same_pass`, teacher, recurrent-head, dual-head, or coefficient ablations until
the shared-final Bank result replicates.

## Acceptance criteria

The handoff is complete when:

- the mixed-pass study and 50 Mi data recipe validate;
- predictor LR grouping and mixed-pass calibration are tested;
- all repository checks pass;
- calibration and provenance are recorded;
- the three primary arms finish at identical token counts;
- the comparison report is reproducible from saved artifacts;
- any positive claim is based on the 50 Mi endpoint and trajectory, followed by
  matched-seed replication rather than the 10 Mi snapshot.
