# Experimental pipeline

This pipeline answers four scoped questions without spending compute on broad
hyperparameter searches:

1. How do dense and sparse content-addressed Tape policies affect long-range
   memory relative to the local-attention backbone?
2. Does sparse Tape still help when writes come from explicit input-only
   `<MEM>` positions rather than periodically selected linguistic tokens?
3. Does Tape provide a useful slow-memory channel beyond a fast recurrent
   channel?
4. Under the chosen TinyMistral training regime, how do Tape and adaptive
   Recirculation compare at similar data and dollar budgets?

All Tape arms use a 32-record bank, readers after decoder layers 3 and 7,
sequence-anchored RoPE, an identity-initialized writer, and zero-initialized
reader output projections. The three write policies are dense, periodic C32,
and explicit-memory-token C32. The explicit `<MEM>` arm uses `write_only`
visibility so the control position can affect later tokens only through Tape.
The active hybrid uses periodic C32 Tape with adaptive Recirculation as its fast
channel. The locked Recirculation placement is source layer 6 to destination
layer 3. Layer indices are zero-based. These are good-enough defaults, not
claims of optimal spacing or placement.

FBT and MemoryAdd remain implemented and covered by architecture correctness
tests for historical compatibility, but both are retired from the active
research program: no wiring, smoke, efficiency, cloud, or confirmation run is
scheduled for either model. The related TapeAddHybrid is likewise not an
active experiment.

## Shared pass protocol

Phase A samples K=2 on 90% of batches and K=3 on 10%:

```yaml
pass_schedule:
  - probabilities: {2: 0.9, 3: 0.1}
ntp_pass_loss_weights_by_k:
  2: [0.0, 1.0]
  3: [0.0, 0.0, 1.0]
```

This estimates `0.9 L2 + 0.1 L3` with 2.1 average passes rather than running
three passes on every batch. Phase B retains a first-pass objective:

```yaml
ntp_pass_loss_weights_by_k:
  2: [0.1, 0.9]
  3: [0.1, 0.0, 0.9]
```

## Stage 0: implementation gates

Directory: `stage_0_implementation_gates/`.

Complete the architecture tests, full suite, study verification, source-tree
cleanliness check, and a wire-only construction of Stage 1. Do not begin a
training trajectory from an uncommitted source snapshot.

## Stage 1: local frozen-backbone wiring

Directory: `stage_1_wiring/`.

Train only architecture-added parameters on MPS. Every arm consumes all
5,242,880 training tokens in `data/dolmino/wiring_2048` exactly once. The
default Phase-A added-parameter learning rate is `1e-4`, including adaptive
Recirculation. Use the final checkpoint rather than selecting the best
intermediate validation score. The downstream stages use `init_from`, not
`resume_from`, so they start new optimizer, sampler, RNG, and pass-scheduler
trajectories.

Local configs set `checkpoint_keep_last: 1` to reduce disk use. This retains no
fallback generation; copy completed wiring checkpoints to durable storage
before deleting local run directories.

A wiring checkpoint is accepted only if all losses are finite, K=3 does not
collapse, reader outputs remain bounded, and the relevant added parameters
receive gradients. Tape is expected to update its output projections on the
first optimizer step; Q/K/V and writer gradients become nonzero after those
projections move away from zero.

## Stage 2: local Phase-B smoke

Directory: `stage_2_local_smoke/`.

Initialize from the canonical Stage-1 checkpoints and train adaptive
Recirculation, all three Tape policies, and Recirculation–Tape for 1M tokens
with the full backbone differentiable. This stage checks stability and
integration only. It is not a final model comparison. These local configs also
retain one checkpoint generation.

Do not proceed with a model that has non-finite gradients, persistent pass-2
regression, K=3 collapse, or recurrent continuation failure.

## Stage 3: cloud pilot

Directory: `stage_3_cloud_pilot/`.

The configs use the document-disjoint `data/dolmino/pilot_2048` artifact and
have 10M-token endpoints, but the first invocation stops at 5M via
`--until-unique-tokens 5242880`. This makes the pilot checkpoint the first
confirmation seed if the arm is promoted; it can resume to the declared 10M
endpoint without changing its trajectory. At 10M, every arm has consumed the
10,485,760-token training split exactly once. The pilot recipe skips the full
5M wiring slice, so pilot training is disjoint from both wiring training and
the shared held-out validation split.

Before paid execution, run CUDA qualification and change hardware batch fields
only if the same change is applied to all directly compared arms. Record
linguistic tokens/s, pass-position compute, peak VRAM, instance runtime, and
dollars. The first 5M pilot includes six arms: Vanilla, adaptive Recirculation,
dense Tape, periodic-C32 Tape, explicit-`<MEM>`-C32 Tape, and the
periodic-C32 Recirculation hybrid. Cloud configs retain two checkpoint
generations.

At the 5M gate, run pass-depth, recurrent-inference, and memory-intervention
diagnostics. Each Tape policy is analyzed independently. A Tape policy is
eligible for promotion only if real Tape performs better than its zero or
mismatched-memory interventions on at least one long-range measure. Promote at
most one Tape policy, using the predeclared long-range result first and cost as
the tie-breaker. At most one hybrid is promoted. A hybrid is eligible for the
slow-memory claim only if its real Tape channel improves over its
recurrent-only intervention; choose between eligible hybrids using the
predeclared long-range result and then cost.

## Stage 4: selected confirmation

Directory: `stage_4_confirmation/`.

Resume the promoted Stage-3 seed to 10M. Then execute the two additional-seed
candidate configs only for Vanilla, the selected Tape policy, the selected
hybrid, and adaptive Recirculation. The preparation command takes the Tape
selection and fixed Recirculation choices, so no execution settings need to be
edited after observing pilot results.

The primary evaluation uses at least 256 fixed validation blocks, controlled
retrieval/state-tracking lags of 32 through 1024, K=1 through K=8 pass-depth
stability, exact-versus-collapsed recurrent continuations, and causal memory
interventions. Report paired block-bootstrap confidence intervals and make
claims conditional on the shared canonical wiring checkpoint unless Phase A is
later repeated independently.

## Compute rule

Local compute pays for all wiring and 1M-token integration checks. Cloud spend
is reserved for qualification, the 5M pilot, and promoted confirmation arms.
Do not run spacing, reader-layer, controller-placement, learned-write, or broad
learning-rate sweeps unless a locked default fails an explicit acceptance gate.

The $50 cloud ceiling is allocated before execution:

| Use | Maximum spend |
| --- | ---: |
| CUDA memory/throughput qualification | $3 |
| Six-arm 5M pilot | $14 |
| Promoted seed-1337 continuation and two additional seeds | $24 |
| Final diagnostics | $4 |
| Storage, interruption, and rerun reserve | $5 |

After qualification, record cost per million linguistic tokens from measured
throughput and shorten confirmation endpoints uniformly if the $24 ceiling
would be exceeded. A failed or interrupted pilot does not borrow from the
reserve without an explicit decision recorded in the Stage-3 results notes.
