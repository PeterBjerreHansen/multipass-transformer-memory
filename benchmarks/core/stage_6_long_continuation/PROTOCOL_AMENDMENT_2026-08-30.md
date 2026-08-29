# Stage 6 protocol amendment — 2026-08-30

This amendment records corrections discovered after the two long runs began.
It does not change either live training trajectory.

The live training checkouts must remain on their recorded source identity
(`d7456cf5d74bc756aef52e2d55184096b744b8a5` at launch). Do not pull this
amendment into a running or resumable training directory. If a run is
interrupted, resume it with the recorded source; run corrected evaluations from
a separate checkout. This preserves the checkpoint source-provenance contract.

## Data status

The validation split inside `gpu_2048_long_2p5b` is a deterministic monitoring
split, but it is not independent of all earlier training. The staged
materializer advances each source stream by the requested validation length and
then by `train_skip_tokens`. Because the wiring and long artifacts used
different validation budgets, much of the long artifact's validation stream
also appeared in the wiring training stream. These NLL values are useful for
within-run convergence and numerical monitoring, but they must not be described
as held-out evidence or used as the primary model-selection result.

Final claims use `data/dolmino/stage_6_evaluation_2048`, whose validation stream
starts after the complete Stage-6 source range. It must pass
`scripts/verify_data_disjointness.py` against every training and wiring artifact
used by either arm. The artifact path and manifest hash must be recorded by the
evaluation command.

## Evaluation lanes

1. **Validation NLL:** exact full-sequence teacher forcing at K=1 through K=8.
   K is pass depth. There is no collapsed recurrence or generation in this
   diagnostic.
2. **Candidate-ranking capability suite:** `evaluation/suites/full.yaml` ranks
   supplied answer candidates by conditional log-likelihood. With feedback
   decoding, each observed candidate token advances the live feedback state
   once before the next candidate token is scored. This is teacher-forced
   candidate scoring, not free generation.
3. **Long free generation:** `generation_math.yaml` (and the isolated
   `generation_code.yaml` suite when appropriate) performs K-pass prompt
   prefill followed by one live decode update per generated token. Prompt K and
   continuation decode mode are independent axes. The primary memory-model
   mechanism is feedback decoding; standard decoding is an ablation. Vanilla
   supports only K=1 standard decoding.

Every retained evaluation must identify its checkpoint hash, normalized config
hash, evaluator source hash, suite hash or data-manifest hash, package versions,
seeds, prompt K, and decode mode, and must retain task-level sample evidence.
Initialized weights are permitted only as an explicitly labelled time-zero
baseline.

The evaluator explicitly disables the tokenizer's training-time fixed padding
and truncation. Earlier Stage-5 task JSON did not do this and can contain empty
candidate continuations with zero scores; those files are invalid and must not
be used as capability baselines.

## Interpretation

The two arms are equal in linguistic-token dose, optimizer-batch size, seed,
backbone learning-rate schedule, and data recipe. They are not equal in FLOPs,
wall-clock compute, initialization history, or parameter count. Consequently,
report both equal-token capability and measured compute; do not call the result
an equal-compute comparison.

The backbone LR was selected by a small diagnostic sweep with a fixed
added-module LR. It is a heuristic choice for this continuation, not an
architecture-wide hyperparameter optimum.
