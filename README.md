# Distributed Feedback Memory for Multi-Pass Transformers

This repository compares feedback memory mechanisms retrofitted into TinyMistral.
Tokens read strictly earlier positions. They never read their own same-position feedback state.

## Current pipeline

The active frozen studies are:

1. [LR qualification](benchmarks/development/frozen_backbone_lr_qualification/README.md):
   completed 48-arm qualification across four rates and one or two sites. Every
   architecture/site group selected `added_learning_rate: 1.0e-3`.
2. [Frozen comparison tiers](benchmarks/development/frozen_backbone_comparison/README.md):
   nested small, medium, and large 100M-token arm sets. They use the same
   protocol and differ only in the number of mechanisms tested. The small tier
   is the primary development run.

The small tier contains a No-memory Adapter, projected-residual feedback,
Recirculation feedback, and aligned-GQA destination-gated Memory Attention at
the fixed `[3, 7]` reader layout. Medium adds stride-8 retention and a
zero-output residual-attention control; large adds the remaining aligned-reader
fusion controls.

The backbone stays frozen throughout these runs. Training uses 2048-token blocks
and K=2/K=3 final-pass loss. Routine validation uses K=4 and retains per-pass scores.
The main runs enable one full-block BOS-only feedback check at about 5M, 20M and 100M.
The LR sweep leaves feedback checks off. See the study pages for exact counts and cadences.

The runner does not select learning rates, run full-split validation, or execute downstream suites automatically.
The clean target-GPU preflight and LR qualification are complete; the tiered
100M trajectories remain planned and must start fresh from the pretrained
checkpoint.
Pass-indexed memory interventions and exact-versus-Live-Feedback diagnostics
are implemented separately from routine validation.

The future unfrozen experiment must start fresh from the pretrained checkpoint.
It needs a separate protocol and LR qualification, preceded by the requested grill session.
See the [development plan](docs/DEVELOPMENT_PLAN.md).

## Documentation

Start with the [documentation map](docs/README.md).
It identifies the authoritative architecture, data, training, evaluation and cloud guides.
The [cleanup ledger](docs/CLEANUP_STATUS.md) preserves the review findings and their resolution.

There is intentionally no central `configs/` directory.
Runnable YAML files live with their study or asset. Each scientific study owns a `STUDY.yaml`.
Raw checkpoints and telemetry belong under the owning study's `results/<arm>/` directory and remain ignored by Git.

| Location | Contents |
| --- | --- |
| `src/tiny_mistral/` | Vendored TinyMistral backbone |
| `src/tiny_mistral_mptt/` | Memory variants, training, inference and evaluation |
| [benchmarks](benchmarks/README.md) | Controls, scientific studies and engineering measurements |
| [data](data/README.md) | Pinned preparation recipes. Generated artifacts are local. |
| [evaluation](evaluation/README.md) | Shared evaluation contract and downstream suites |
| [scripts](scripts/README.md) | Command-line entry points |

## Setup and checks

Python 3.10–3.13 is supported. Run commands from the repository root.

```bash
uv sync --extra data --extra eval
make check
```

To use an existing environment with the required dependencies:

```bash
PYTHONPATH=src pytest -q
```

Prepare and verify the active artifact before target-GPU preflight:

```bash
uv run python scripts/prepare_data.py --config data/dolmino/gpu_2048/config.yaml
uv run python scripts/verify_data.py data/dolmino/gpu_2048
```

See [data preparation](docs/DATA.md) and [cloud preflight](docs/CLOUD.md).
Local tests do not establish CUDA fit or throughput.

## Historical results

The completed eight-arm screen and Stage-6 continuation protocol remain under
[the historical staged pipeline](benchmarks/historical/staged_pipeline/README.md).
The [Stage-5 result table and commentary](benchmarks/historical/staged_pipeline/stage_5_cloud_100m/results/README.md)
retain the original findings and split-overlap caveats.
These are not results for the restructured frozen comparison.
Earlier unauditable downstream JSON is not valid capability evidence.

Paper replay/BPTT and the 1024-token studies are deleted. The active recurrent
models use the shared late writer and support adaptive recirculation mixing.
FBT and explicit memory-token attention remain as
[preserved reference source](historical/implementations/feedback/README.md), outside
the active runtime. Memory Attention can optionally add the current late
recurrent-memory pathway; see [its configuration](docs/MEMORY_ATTENTION.md#7-optional-recurrent-memory-hybrid).
