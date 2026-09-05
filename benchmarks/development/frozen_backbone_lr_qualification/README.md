# Frozen-backbone learning-rate qualification

This planned qualification selects the Phase-A learning rate for every
architecture/site setting in the frozen-backbone wiring comparison before its
trajectory is interpreted.
The backbone remains frozen throughout each run; only the added writer and
merger or Memory Attention parameters are optimized. The recurrent arms use
late emitted memory with projected-residual and adaptive-recirculation mergers.
This qualification will use the regenerated clean 2,048-token `gpu_2048` artifact and nominal
65,536-token optimizer batch as the comparison study. The dense-and-strided
arms use a 4x8 microbatch/accumulation split to fit the A6000; the other arms
use the 8x4 default split.

The primary dense mechanisms are qualified separately at one site `[3]` and
two sites `[3, 7]`. The attention extensions retain matching one-site and
two-site layouts; the stride-length ablation reuses the selected two-site
Strided Memory Attention rate because stride changes retention rather than
trainable parameterization. Complete the bounded real-trainer GPU preflight
before this sweep; token-zero evaluation and expanded memory diagnostics are
not prerequisites. Feedback NLL remains disabled here; the selected
5M/20M/100M feedback schedule applies only to the main 100M runs.
In particular, qualification does not run the BOS-only full-block Live Feedback
evaluator or the intervention/fidelity diagnostics. Its only scheduled quality
evaluation is the routine parallel K=4 validation described below.

## Protocol

| Config field | Value |
| --- | --- |
| `phase` | `A` |
| `max_unique_tokens` | `5013504` |
| `batch_size` | `8` (dense-and-strided arms: `4`) |
| `grad_accum_steps` | `4` (dense-and-strided arms: `8`) |
| `eval_passes` | `4` |
| `eval_batches` | `64` |
| `eval_every_tokens` | `1048576` |
| `feedback_eval_at_tokens` | `null` |

Every arm consumes exactly 5,013,504 unique linguistic tokens. With 2,048-token
blocks and either 8x4 or 4x8 batching this is 77 optimizer updates, with the
final update using the remaining partial accumulation. The pass schedule
is the shared wiring schedule: K=2 with probability 0.9 and K=3 with probability
0.1, with final-pass-only NTP loss weights. Validation uses deterministic K=4
whole-block pass-depth NLL on 64 held-out blocks; generation is not part of this
qualification.

The candidate grid is the same for every mechanism:

```text
3e-5, 1e-4, 3e-4, 1e-3
```

This is 48 independent fresh runs: twelve architecture/site configurations times
four constant added-parameter learning rates, with one seed (`1337`) per
setting. Each run starts from the common pretrained checkpoint, not another
qualification run. The total tuning dose is 240,648,192 linguistic training
tokens across the 48 runs; they reuse the same artifact, so this is not that
many distinct corpus tokens. The backbone is frozen and has no trained
optimizer group in this sweep.

Validation is recorded every 1,048,576 tokens and at the final 5,013,504-token
state. Training records are 327,680-token aggregates. A short run is enough to
reject unstable rates and identify large optimization differences, but it is
not evidence that the selected rate is globally optimal. If candidates remain
close, extend the finalists before locking the 100M wiring study.

Run all 48 arms sequentially on the target CUDA host:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_lr_qualification \
  --skip-wire
```

Choose each mechanism's rate using final-pass K=4 validation NLL plus the
minimal finite-loss/no-catastrophic-instability guardrail. The long comparison
may use different selected rates per mechanism; retain `added_learning_rate` as
an explicit allowed difference in the long comparison and report the qualification budget.

The 2,048-token qualification supersedes the earlier 1,024-token rate sweep.
The earlier qualification was invalidated with the padded artifact. Repeat the
48 fresh arms after regenerating and verifying the clean artifact; do not carry
the earlier selections into the reset comparison without requalification.
