# Distributed Feedback Memory for Multi-Pass Transformers

This repository compares feedback memory mechanisms retrofitted into TinyMistral.
Tokens read strictly earlier positions. They never read their own same-position feedback state.

## Experiments

The program compares frozen-backbone wiring with full-model adaptation:

- [Frozen LR qualification](benchmarks/development/frozen_backbone_lr_qualification/README.md)
- [Frozen comparison tiers](benchmarks/development/frozen_backbone_comparison/README.md)
- [Vanilla backbone LR qualification](benchmarks/development/unfrozen_lr_qualification/README.md)
- [Unfrozen scaling comparison](benchmarks/development/unfrozen_scaling_core/README.md)

Each study owns its protocol, launch requirements and status. The
[research plan](docs/RESEARCH_PLAN.md) defines the hypotheses and claim limits.
Frozen and unfrozen results can reveal whether backbone adaptation changes the
ranking of feedback mechanisms.

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
