# Command-line entry points

Run commands from the repository root. Use `uv run python scripts/<name>.py --help`
for each Python command's arguments. Scientific settings live with their owning study.

## Setup and correctness

| Command | Purpose |
| --- | --- |
| `download_model.py`, `verify_model.py` | Fetch or verify the pinned TinyMistral checkpoint |
| `compare_to_hf.py`, `compare_to_hf_layers.py`, `compare_to_hf_inputs_embeds.py` | Compare the vendored backbone with the pinned Transformers oracle |
| `prepare_data.py`, `verify_data.py` | Materialize or verify packed data |
| `verify_data_disjointness.py` | Check complete BOS-delimited documents across artifacts |
| `verify_study.py` | Validate active manifests and declared comparison differences |
| `smoke_mps.py` | Run local Apple-hardware checks |

See [data](../docs/DATA.md), [study organization](../benchmarks/README.md) and
[validation gates](../docs/VALIDATION.md).

## Training and operation

| Command | Purpose |
| --- | --- |
| `train.py` | Train one config, optionally restoring a durable trajectory |
| `run_study.py` | Verify a study, run forward/backward checks, then train selected arms sequentially |
| `cloud_preflight.py` | Check CUDA, input integrity, source/run compatibility and persistent storage |
| `start-and-watch` | Start or observe a remote run, transfer verified outputs, then apply requested Verda cleanup |
| `run-cloud-study` | Apply that lifecycle to selected arms of a locked study |

`run_study.py --wire-only` does not perform an optimizer step.
It does not replace the [real-trainer preflight](../docs/CLOUD.md#pre-training-checks).
Neither cloud wrapper performs that qualification automatically.
The current planned studies cannot run through the locked-study wrapper.

See [training and recovery](../docs/TRAINING.md) and [cloud operation](../docs/CLOUD.md).
These guides define resume and cleanup behavior. The script index does not duplicate those contracts.

## Evaluation

| Command | Purpose |
| --- | --- |
| `evaluate_nll.py` | Final parallel-pass NLL, or full-block BOS feedback with `--forward feedback` |
| `evaluate_pass_depth.py` | Parallel NLL and hidden-state changes through the requested K |
| `evaluate_memory_interventions.py` | Real/zero/mismatched memory for one K=1-to-K=2 transition |
| `evaluate_recurrent_inference.py` | Same-checkpoint exact, feedback and standard continuation comparison |
| `evaluate_lm_harness.py` | Candidate scoring or generation with explicit prefill/decode choices |
| `evaluate_parameter_drift.py` | Backbone and added-parameter changes relative to compatible reference weights |
| `generate.py` | Ordinary pretrained-backbone generation, not trained feedback-model generation |

The checkpoint evaluators require explicit weights or `--initialized-baseline`.
Parameter drift requires its checkpoint and reference.
`generate.py` loads the pretrained backbone directly.
See [evaluation](../evaluation/README.md) for defaults, precision, target sets and result identity.
See [cached inference](../docs/RECURRENT_INFERENCE.md) for the lower-level feedback interface.

## Engineering measurements

- `benchmark_training_efficiency.py`: synthetic training with real optimizer steps.
- `benchmark_inference_efficiency.py`: synthetic full-pass and cached-continuation timing.
- `estimate_training_flops.py`: dominant-matmul estimates from a suite or study.
- `select_cuda_batch.py`: choose a candidate from an engineering batch report.

Engineering grids are not scientific arms.
The general training grid includes legacy recirculation, not both active recurrent mergers.
The inference script uses the five current arms but does not yet time the production BOS NLL evaluator.
See [efficiency measurements](../benchmarks/efficiency/README.md) and
[the retained A6000 report](../benchmarks/development/inference_efficiency/README.md).
