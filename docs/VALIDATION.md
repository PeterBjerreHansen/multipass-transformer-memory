# Validation gates

Correctness gates are part of the research protocol. Experimental conclusions
belong under `benchmarks/`; this file describes reusable invariants that must
remain green.

## Vanilla substrate

The vendored backbone targets `M4-ai/TinyMistral-248M-v3`. Provenance is in
`UPSTREAMS.md`; `VANILLA_SOURCE.sha256` guards the vendored source.

Memory Attention work adds one documented substrate capability: an optional boolean
self-attention K/V-validity mask. Ordinary runs use all-valid keys and must retain
vanilla numerical behavior. Reference, local O(TW), and FlexAttention masks are
tested for parity, including an all-masked query row returning exact zero rather
than NaN/uniform leakage.

The sparse-SWA control adds an opt-in multiresolution mask to selected layers.
The ordinary all-local path remains unchanged. Compact local attention,
reference masking, and cached explicit-position retention must agree.

## Multipass/Memory Attention gates

The suite must enforce:

- ordinary pass 1 matches the SWA Transformer backbone path;
- FBT exact cached inference matches full-prefix multipass recomputation;
- retired checkpoint controls retain their focused compatibility tests;
- adaptive recirculation starts at the configured fixed mixture;
- adaptive recirculation Phase A freezes the TinyMistral backbone and trains
  only its coefficient controller;
- dense Memory Attention and strided C1 are identical with matching weights;
- multiscale Memory Attention reduces to Dense Memory Attention when `S=0` and Strided Memory Attention when
  `D=0`, and uses one softmax over a non-overlapping union otherwise;
- zero-initialized Memory Attention is an exact SWA Transformer fixed point at all pass depths;
- Memory Attention reader allocation and projected caches match `memory_layers`;
- memory RoPE retains original linguistic write/query positions through cached eviction;
- strided and MEM writes are strict-past;
- memory window counts records and empty/invalid records return finite exact-zero
  attention contributions;
- Phase A freezes pretrained parameters and trains only added parameters;
- memory-token Phase A preserves pass-1 autograd for the added MEM embedding,
  which receives Memory Attention-mediated gradients after zero-output reader activation;
- pass weights and pass-count scheduling are deterministic and checkpointable.

## Explicit MEM loss/attention gates

For `A <MEM> B`:

- MEM input ID is V while the LM output dimension remains V;
- A targets B and the MEM position has `ignore_index`;
- direct LM gradient at MEM's logits is exactly zero;
- perturbing ignored MEM-position logits cannot change the language loss;
- after reader output activation, the MEM embedding receives nonzero Phase-A
  gradient through recurrent/Memory Attention pathways;
- `visible` permits a local MEM-to-future self-attention dependency;
- `write_only` permits MEM to read preceding context but prevents MEM from being
  used as self-attention K/V;
- cached write-only key validity preserves the MEM physical position.

## Cached/recurrent gates

- exact incremental K-pass equals full-prefix recomputation for multiple K;
- snapshot-before-update prevents same-position feedback leakage;
- recurrent prefill starts from the exact K-pass boundary;
- the first recurrent continuation transition equals exact K-pass;
- Memory Attention state remains chronological and bounded;
- write-only cache validity persists across decode;
- K=1 remains the SWA Transformer cached boundary;
- Strided Attention adds no parameters, changes only selected self-attention masks,
  and cached decoding equals full-sequence execution.

## Training/recovery gates

The spot-safe trainer is tested for:

- two durable checkpoint generations by default, with explicit one-generation
  retention supported for local runs;
- a new generation being verified before `latest.json` advances;
- corrupt-newest fallback to the previous generation;
- incomplete `.tmp` files being ignored;
- metrics repair back to checkpointed progress;
- source-code/environment identity checks on resume;
- interruption/resume produces the same final model, optimizer, sampler, and
  counters as uninterrupted training.

Data preparation is tested for deterministic source allocation, checksums, and
recorded source-balanced offsets. Related recipes must share validation
settings, avoid overlapping training slices, and match their declared budgets.

## Study and hardware gates

`make check` runs pytest, byte-compilation, study-manifest verification, and
`git diff --check` when Git metadata is available. On Apple hardware also run
`scripts/smoke_mps.py`.

Before a serious CUDA campaign, run the K=2 batch qualification with
`grad_accum_steps=1`. A larger selected microbatch changes optimizer-batch size
unless accumulation is adjusted and therefore requires scientific qualification.
For write-only MEM, CUDA FlexAttention/reference parity must remain green before
paid quality runs.
