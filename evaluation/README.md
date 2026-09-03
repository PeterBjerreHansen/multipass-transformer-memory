# Evaluation contract and task suites

Training validation and standalone evaluation share execution precision, target
scoring and token-weighted aggregation. Experiment configs supply defaults;
explicit command-line arguments override them. There are no named evaluation
profiles.

## Parameters, not bundled modes

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--passes` | `eval_passes` | Maximum parallel depth; K=4 in the active frozen studies |
| `--prefill-passes` | `eval_prefill_passes`, otherwise `eval_passes` | Passes over the actual prompt only |
| `--decode-mode` | `eval_decode_mode`, otherwise feedback for supported models and standard for single-pass models | One-stream continuation policy, independent of K |
| `--autocast-dtype` | Experiment `autocast_dtype` | `config`, explicit `float32` (no autocast), or `bfloat16` |
| `--device` | Experiment `device` | Execution device; does not modify checkpoint compatibility settings |
| `--max-blocks` | All blocks for standalone packed evaluators | Prefix-subset limit, **not** a batch size |

The trainer uses `eval_batches` for its block limit and `eval_every_tokens` for
its cadence. The main comparison and LR sweep have different cadences.
See [the study protocols](../benchmarks/README.md). To reproduce a routine check
standalone, use the same weights, artifact, precision, K and block limit.
Omitting the standalone limit still selects the full split. Blocks are evaluated
one at a time; multi-block batching remains a later addition.

A K=1 BOS-only prompt uses the ordinary feedback implementation. It is not a new
mode. Downstream tasks normally use K=4 actual-context prefill followed by one
feedback step per observed candidate token or generated token. A vanilla model
uses K=1 standard decoding; unsupported requests fail, without fallback.

## Precision

`dtype` sets parameter storage. The active CUDA configs use FP32 parameters and
BF16 autocast for eligible operations. Ordinary AdamW state therefore remains FP32.
BF16 compute is not weight quantization.

```yaml
device: cuda
dtype: float32
autocast_dtype: bfloat16
```

The CLI inherits `autocast_dtype` from the experiment. Use
`--autocast-dtype float32` to disable autocast or `bfloat16` to request it explicitly.
Overrides do not modify the training config used for checkpoint compatibility.
Direct Python evaluators have no experiment config. Their default is no autocast.

The shared evaluation context restores model training mode after success or failure.
It calculates each cross-entropy reduction in FP32 and accumulates the returned
loss sums and target counts across blocks. Reports record actual parameter
dtypes, autocast, device and loss dtype. Compare results at matched precision.

Selected-checkpoint feedback uses `feedback_eval_autocast_dtype`:
`config`, `float32` or `bfloat16`. It can differ explicitly from routine validation.
A changed precision request gets a separate durable feedback report.

BF16 autocast requires supported CUDA or MPS hardware and runtime.
CPU evaluation of a CUDA/BF16 experiment requires `--device cpu --autocast-dtype float32`.
Unsupported requests fail without a silent precision change.
MPS normally uses FP32. MPS BF16 remains an engineering option until qualified on the target host.
The efficiency runner records unavailable modes as unsupported.

The trainer accepts only BF16 or disabled autocast. Enabling autocast requires
`dtype: float32`. FP16 training has no validated loss-scaling policy.
Pure BF16 parameter storage is not the active training protocol.

## Packed NLL and diagnostics

By default, `scripts/evaluate_nll.py` returns the final-pass view of the same parallel
evaluator used by `scripts/evaluate_pass_depth.py` and trainer validation.
Pass-depth evaluation retains every pass from K=1 through the resolved maximum
and hidden-state deltas. It no longer defaults independently to K=8; request
`--passes 8` explicitly for that diagnostic.

Targets come from `model.build_lm_labels`. Ordinary blocks score positions
1 through L-1 as next-token targets: the first token is not predicted and no BOS
is prepended. Input-only MEM slots are excluded while predictions bridge to the
next linguistic token. Loss is summed in FP32, then divided by the number of
scored tokens, globally and per source. Sources with different target counts
are not equally weighted. Empty target selections fail explicitly.

`scripts/evaluate_recurrent_inference.py` compares exact cached K-stream
decoding, feedback decoding, and standard K=1 decoding from the **same checkpoint**.
It scores only the selected continuation after a data-prefix prompt.
`--prompt-tokens 1` takes the first data token; it does not insert BOS. Results
include per-offset and per-horizon target counts, per-source scores and the
actual prompt/continuation lengths. Exact K-pass is a correctness reference,
not the default downstream continuation policy.

New diagnostic keys are `standard_k1_nll`, `standard_k1_nll_by_offset`, and
`recurrent_minus_standard_k1`. These replace the old `vanilla` names, which did
not denote a separately trained vanilla model. Historical JSON is not rewritten.

`scripts/evaluate_memory_interventions.py` still tests one pass-1-to-pass-2
transition with real, zero and mismatched memory. It now uses the shared scorer,
precision context and per-source aggregation. The mismatch donor is the next
block modulo the full artifact length, even when the scored subset is smaller;
this rule is recorded. Use this simple check on early trained snapshots first;
dedicated token-zero evaluation and deeper diagnostics are deferred. General
depth-configurable memory diagnostics remain later work; K=4 will be an
experiment default, not an implementation limit. Zero memory is not equivalent
to bypassing an adaptive recirculation merger, which can still rescale its input.
See [the development plan](../docs/DEVELOPMENT_PLAN.md).

Intervention reports use schema version 3. Hybrid channel keys explicitly name
both channels, for example `zero_recurrent_real_attention` and `real_recurrent_zero_attention`;
the combined `real_memory`, `zero_memory` and `mismatched_memory` conditions are unchanged.
FLOP reports use schema version 2 and `memory_writer`, `memory_reader_projections`,
`memory_reader_products` and `memory_write_positions` fields.
Pure-model field renames do not change the computation. The optional late-memory
hybrid is a new architecture, not a rename of the deleted hybrids. CLI provenance records
its recurrent merger and layer configuration. Older JSON retains its original schema.

## Full-block BOS-only feedback NLL

`evaluate_nll.py --forward feedback` reuses ordinary feedback decoding with
exactly one BOS token and K=1 prefill. It teacher-forces the entire block, resets
state for each block and never samples, crops a block, or runs exact-reference
streams. This selects a scoring protocol, not a new inference algorithm.
`--passes` is fixed to 1 for this request; routine parallel and downstream
prefill defaults remain K=4.

For a 2048-token block, `nll` scores all 2048 tokens, including the first token
conditioned on BOS. `aligned_nll` excludes that first target and scores the same
2047 target IDs as parallel next-token NLL. It still has added BOS context:
aligned targets do not mean identical conditioning or computation. Both scores
include token counts and per-source breakdowns. Existing document-separator
BOS tokens in the artifact are preserved and scored as data.

The decoder consumes BOS plus the first 2047 data tokens, then scores the final
target without consuming it. Thus no extra context position or block truncation
is needed. In memory-token views, the existing insertion cadence is retained;
BOS does not count toward it. Labels remain model-owned: in `A <MEM> B`, A
predicts B and MEM logits are ignored, although MEM is consumed into the cache.

```bash
uv run python scripts/evaluate_nll.py \
  --config <arm.yaml> --checkpoint <snapshot.safetensors> \
  --forward feedback --max-blocks 1 --autocast-dtype float32 \
  --output <feedback-result.json>
```

For trainer integration, use `feedback_eval_at_tokens` to select a subset of
`snapshot_at_tokens`; see [the schedule and recovery contract](../docs/TRAINING.md#selected-checkpoint-feedback-validation).
`feedback_eval_max_blocks` defaults to one complete prefix block. All arms must
use the same artifact, prefix length and feedback precision. One block is a
diagnostic, not a statistically reliable ranking or full-split evaluation.

The [main frozen protocol](../benchmarks/development/frozen_backbone_comparison/README.md)
selects the checkpoint milestones. The LR sweep leaves feedback off.
Reports retain requested thresholds and actual optimizer-boundary token counts.

## Downstream suites

Reusable `lm-evaluation-harness` task suites are separate from training configs:

- `suites/quick.yaml`: small development sanity battery.
- `suites/full.yaml`: candidate ranking by conditional log-likelihood, not free generation.
- `suites/generation_math.yaml`: long free-generation math evaluation.
- `suites/generation_code.yaml`: code generation with program execution; run only in isolation.

The standard and feedback scorers share causal context/continuation truncation.
Only retained continuation targets contribute. Empty context gets an explicit
BOS from the harness; candidate tokens are teacher-forced, not sampled.
The final scored target need not be consumed into the cache.

The adapter records scored-token operations and generated-token counts. These
are execution totals, including repeated candidate requests, not counts of
unique documents. The current text adapter does not insert a physical MEM
schedule; memory-token downstream evaluation remains unsupported in that sense.

## Result identity and comparison

Standalone commands require either `--checkpoint` or an explicitly labelled
`--initialized-baseline`. For configs with `init_from`, time zero includes those
wired weights but starts before the new optimizer trajectory.

Retained JSON includes checkpoint/config/source hashes, seeds and package
versions. Packed results add the split, manifest hash, declared artifact hashes,
exact prefix selection, physical/linguistic lengths, control-slot cadence,
resolved computation policy, precision and scored-token counts. Declared binary
hashes identify the artifact; routine validation does not re-hash multi-GB token
files. Verify artifacts separately before comparing runs.

Trainer records identify live weights by run, segment, optimizer step and token
count, rather than claiming that an unsaved state is a checkpoint. Downstream
results retain suite/tokenizer hashes, task configs, raw samples and available candidate
margins. The same contract can be used by a future fresh unfrozen experiment;
freezing is not an evaluation mode.

Older standalone results may have used FP32 despite a BF16 training config.
Do not relabel them as BF16 or combine unmatched settings.

Write retained output beside the relevant study/checkpoint. Use
`--evaluation-data-dir` to select a separate artifact for final claims; the
monitoring split does not become independent evidence merely by changing the
command.

## Cost and remaining work

The [A6000 report](../benchmarks/development/inference_efficiency/README.md)
supports routine parallel K=4 checks and selected post-snapshot feedback
diagnostics, initially one fixed full block per arm (`--max-blocks 1`).
The combined exact-vs-feedback cost includes reference decoding and diagnostics;
it is not feedback-only cost. BF16 is not uniformly faster across these paths.

See [training](../docs/TRAINING.md) for snapshot recovery and scheduling.
The [development plan](../docs/DEVELOPMENT_PLAN.md) tracks batching, expanded
interventions and the separate unfrozen study.
