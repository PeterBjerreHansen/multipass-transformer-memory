# Experimental pipeline

The reported eight-arm study is specified across stages 0 through 5. It compares
vanilla TinyMistral, adaptive Recirculation, three Bank write policies, the
Recirculation–Periodic Bank hybrid, Multiscale Bank, and Sparse SWA:

- Multiscale Bank tests short- and long-range access to previous-pass states
  without a recurrent channel.
- Sparse SWA tests the same pattern over ordinary current-pass token states,
  without recurrence, Bank cross-attention, or added parameters.

These controls separate three possible causes of the reported hybrid gain:
broader attention, access to previous-pass representations, and recurrence.

## Locked architecture defaults

The original Bank models use a 32-record capacity, readers after decoder layers
3 and 7, sequence-anchored RoPE, an identity-initialized writer, and
zero-initialized reader outputs. Their write policies are dense, periodic C32,
and write-only memory-token C32. Adaptive Recirculation routes source layer 6
to destination layer 3. The hybrid places readers after layers 4 and 7.

Multiscale Bank uses the hybrid's reader placement with 32 recent dense records
and 32 older C32 records. Sparse SWA augments self-attention after layers 3 and
7 with 32 older C32 tokens. These are controlled defaults, not claims of
optimal spacing or placement.

FBT, MemoryAdd, and BankAddHybrid remain only for historical compatibility.
They are not active study arms.

## Pass protocol

Multipass arms sample K=2 on 90% of batches and K=3 on 10%. Phase A applies
loss only to the final pass. Phase B gives the first pass weight 0.1 and the
final pass weight 0.9. This costs 2.1 average passes per batch.

Sparse SWA is a one-pass Transformer. It uses K=1 and has no Phase A because it
adds no trainable parameters.

## Existing stages

- Stage 0 defines architecture, causality, cache, study, and substrate
  gates.
- Stage 1 wires added pathways with the backbone frozen on the 5,242,880-token
  `wiring_2048` slice.
- Stage 2 provides 1M-token local Phase-B integration checks.
- Stages 3 and 4 retain the cloud pilot and promotion protocol.
- The core Stage-5 study contains the locked 100M-token eight-arm continuation.
  Its six original arms are complete; Multiscale Bank and Sparse SWA are the
  remaining runs.

The original manifests and configs remain in their stage directories as
protocol records.

## Attention-control extension

### Stage 1: Multiscale Bank wiring

Run only the new wiring arm:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_1_wiring \
  --arm bank_multiscale_wiring_5m
```

Accept the checkpoint only if losses are finite, K=3 remains stable, and Bank
reader, writer, and projection gradients activate as expected. Copy the final
checkpoint to durable storage before removing local artifacts.

### Stage 2: local integration

Run the two new smoke arms:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/stage_2_local_smoke \
  --arm bank_multiscale_smoke \
  --arm sparse_swa_smoke
```

These runs test training integration and stability. They are not quality
comparisons.

### Stage 5: 100M-token attention controls

After the local gates and CUDA qualification pass, run the configs under
`benchmarks/core/stage_5_cloud_100m/`. Multiscale Bank initializes from its
Stage-1 checkpoint. Sparse SWA starts from the canonical TinyMistral checkpoint.
Both use the same Phase-B data slice and optimizer batch as the six completed
Stage-5 arms.

Compare their validation and efficiency results against the locked Stage-5
results. Do not treat Multiscale Bank and Sparse SWA as a tightly matched pair:
they differ in pass count, parameter count, and initialization because those
differences define the architectures being controlled.

## Execution gates

Before local training:

- run `make check` and `scripts/smoke_mps.py`.
- verify the model and `wiring_2048` artifacts.
- use a clean, committed source snapshot for a retained trajectory.

Before paid CUDA training:

- materialize and verify `data/dolmino/gpu_2048_staged`.
- preserve `batch_size=1` and `grad_accum_steps=1` unless a separate protocol
  decision qualifies a change.
- run CUDA BF16, throughput, memory, and cloud preflight checks.
- keep two durable checkpoint generations.

Do not add spacing, reader-placement, or broad learning-rate sweeps unless a
locked default fails a stated gate.
