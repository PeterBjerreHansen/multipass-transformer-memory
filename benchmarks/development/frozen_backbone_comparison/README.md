# Frozen-backbone comparison tiers

This directory contains three nested frozen-backbone studies. They use the
same 100M-token protocol and differ only in how many mechanisms are tested:

| Tier | Arms | Scope |
| --- | ---: | --- |
| [`small`](small/STUDY.yaml) | 4 | Primary adapter, recurrent, and aligned destination-gated attention comparison |
| [`medium`](medium/STUDY.yaml) | 7 | Small plus stride-8 retention and a zero-output residual control |
| [`large`](large/STUDY.yaml) | 10 | Medium plus the remaining aligned-reader fusion controls |

The small tier is the primary development run. The tiers are nested, and an
arm appearing in more than one tier has the same
configuration and starts fresh from the common pretrained checkpoint; no tier
continues another tier's weights. The tiers are a breadth-of-models choice,
not a quality-of-evaluation choice.

## Shared protocol

All arms inherit [`protocol_100m.yml`](protocol_100m.yml):

| Config field | Value |
| --- | --- |
| `phase` | `A` |
| `data_dir` | `data/dolmino/gpu_2048` |
| `max_unique_tokens` | `100007936` |
| `batch_size × grad_accum_steps` | `8 × 4 = 32` sequences |
| `added_learning_rate` | `1.0e-3` |
| `eval_passes` | `4` |
| `eval_batches` | `64` |
| `eval_every_tokens` | `3276800` |
| `feedback_eval_at_tokens` | `[5013504, 20021248, 100007936]` |

The effective optimizer batch is 65,536 linguistic tokens. The dense-and-
strided arm uses `4 × 8` on the target GPU when required by memory, preserving
the same effective batch. Every arm uses 2048-token blocks, K=2 with
probability 0.9, K=3 with probability 0.1, and final-pass-only NTP loss.

Durable snapshots and selected feedback checks are:

| Requested tokens | Actual tokens with default batching | Optimizer updates | BOS feedback NLL |
| ---: | ---: | ---: | --- |
| 3,276,800 | 3,276,800 | 50 | No |
| 5,013,504 | 5,046,272 | 77 | Yes |
| 10,027,008 | 10,027,008 | 153 | No |
| 20,021,248 | 20,054,016 | 306 | Yes |
| 50,003,968 | 50,003,968 | 763 | No |
| 100,007,936 | 100,007,936 | 1526 | Yes |

All memory arms use reader layers `[3, 7]`, a 32-record capacity, 16 KV heads,
and RoPE. The primary attention representative uses `aligned_gqa` initialization
and destination-gated fusion. Medium retains an explicit `zero_output` plus
`residual` dense control. Gated arms use controller width 64.
`aligned_gqa` is used only where the arm name says so; it is not a global
default.

The `[3, 7]` layout and stride 8 are fixed development choices informed by
private exploratory checks. They are not claims of globally optimal placement
or retention. This directory does not repeat one-site versus two-site or stride
matrices. It also excludes memory-token models and recurrent-attention hybrids,
which address different architectural questions.

The completed base-family LR qualification favored `1e-3` uniformly. The same
rate is intentionally adopted for aligned-GQA and gated fusion in development;
those additions are not claimed to have an independently optimized rate.

## Evaluation and launch

Routine validation is parallel K=4 NLL on the fixed 64-block prefix. The
selected BOS-only feedback checks run after durable snapshots at approximately
5M, 20M, and 100M tokens, using one complete block. Full-split feedback,
intervention, fidelity, downstream, wall-clock, and FLOP reports remain
standalone diagnostics and are run on selected snapshots rather than during
every validation check.

Run one tier at a time:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_comparison/small
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_comparison/medium
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_comparison/large
```

Use the small tier by default. Medium and large are optional breadth extensions,
not prerequisites for the primary development comparison or locked paper
benchmarks.
