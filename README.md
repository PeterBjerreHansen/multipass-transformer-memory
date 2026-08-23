# TinyMistralMPTT
tl;dr: Replacing the "quasi recurrent" information-transfer patterns used in the recent "jacobi/multi-pass" trained models like the [full bandwidth transformer](https://arxiv.org/abs/2608.08888) and the [recirculation](https://arxiv.org/abs/2608.17981) model with attention seemlingly improves performance. However, combining the "fast recurrent connection" with sparse and long-range attention perhaps works even better. 

I retrofitted a [tiny pretrained LLM](https://huggingface.co/Locutusque/TinyMistral-248M) into doing feedback-style inference by training it with multi-pass Jacobi updates. The new parameters where first wired into the frozen backbone for 5 million tokens, then the whole models where trained on 100M tokens from the [OLMo2 annealing mixture](https://huggingface.co/datasets/allenai/dolmino-mix-1124).

| Model / method | Final validation PPL | Late PPL reduction, 50M → 100M | Total parameters | Relative training FLOPs |
|---|---:|---:|---:|---:|
| Transformer baseline | 7.778 | 0.110 (1.40%) | 248.024M | - |
| Adaptive Recirculation | 7.678 | 0.110 (1.42%) | 253.275M | 2.127x |
| Sparse Memory Attention (with memory-tokens) | 7.616 | 0.133 (1.72%) | 254.321M | 2.187x |
| Sparse Memory Attention (no memory tokens) | 7.599 | 0.113 (1.46%) | 254.320M | 2.122x |
| Dense SWA Memory Attention | 7.534 | 0.112 (1.46%) | 254.320M | 2.133x |
| Recirculation + Dense Memory Attention Hybrid | 7.519 | 0.120 (1.57%) | 259.571M | 2.149x |

The MemoryAttn uses cross-attention over the multi-pass generated "memories" $[m_{t-c}^k, ... ,m_{t-1}^k]$ (where $c$ is the number of retained memory latents and $k$ pass) instead of a only using $m_{t-1}^k$. The sparse attention variant has the memories attended to spread out with a stride of $s$, and the hybrid model combines the recirculation mechanism with cross-attention over sparse memories.

This experiment is riddled with confounders such as differing compute budgets, paramenter-count and starting-point inequality (some mechanisms are more easily slotted into a pretrained model, and the vanilla backbone had no adaptation). Moreover, I simply do not have the computational ressources to provide the exhaustive sweeps and ablations one needs to make convincing optimality arguments. However, as a preliminary investigation, I think there are two notable patterns: 

1. The attention-based mechanisms outperform the recurrent mechanisms almost uniformly. Since replacing recurrent connections with attention has a solid track-record, I'd wager this trick would work at scale aswell. 
2. The sparse attention to far away memories outperformed models with only the recurrent connections to memories. To me this is result slightly favours the notion that the performance-gains reported in the FBT and recirculation papers stem from *mere greater effective depth* rather than *unlocking recurrent computation patterns*. 
3. The recurrent/attention hybrid performs on par with the dense attention model, and so attention over a sparse memory-bank could provide a valuable "slow" (long-range) memory for multi-pass models with only the "fast" (short-term) recurrent pattern. 

### A Note on Eficciency 
`Relative training FLOPs` is normalized to the vanilla K=1 run at 1.000x. It uses the 90% K=2 / 10% K=3 training schedule, so most of the $>2x$ multiplier is because I choose to always run at least two passes. This policy puts more pressure on the memory-adaptations (which is what I wanted to study most), but is presumably computationally suboptimal since single passes will adapt the backbone to the training distribution more efficiently. The measure counts mostly the dominant dense matrix products in the forward and backward passes, and you can see the estimator is `scripts/estimate_training_flops.py`.


## More Ideas, Future Work
1. **Next Memory Prediction** I have a strong and unproven suspicion that something like a [NextLat](https://arxiv.org/abs/2511.05963) auxillary loss with the memories as targets will work nicely. Since there is alot of pressure on the memory latents to be useful *inputs* to the model, they should contain information that is already useful for NTP prediction and so they should be less prone to collapse. In some sense predicting these memories amounts to predicting features that the model will think are useful later. I like this notion that one should want to predict what is itself predictive. 
2. The recurrent/attention hybrid might be a "best of both worlds" type of model, but we really should compare with a purely atten-based model that has both long and short range attention.
3. Most SOTA models still use dense attention in at least a few layers, I suspect that a local SWA attention + dense long-range attention over memories might perform better than placing the dense attention in a normal layer. 
4. I want to properly wire in the [FBT](https://arxiv.org/abs/2608.08888) model next and test out hybrids. However, the process is a little more tricky for FBT as the mechanism is not residual.  

## Repository map
- `src/tiny_mistral/`: validated vendored vanilla TinyMistral implementation.
- `src/tiny_mistral_mptt/`: research architectures, training, evaluation, and inference.
- `benchmarks/`: controls, development/core studies, historical evidence, and
  engineering efficiency measurements.
- `data/`: deterministic dataset recipes; generated artifacts are local/ignored.
- `evaluation/`: reusable evaluation-suite definitions.
- `docs/`: architecture, data, training, inference, cloud, and validation contracts.

There is intentionally no central `configs/` directory. Runnable settings live
with the study or asset that owns them. Development/core studies use
`STUDY.yaml` for the scientific question and comparison structure; runnable YAML
files remain the execution source of truth.

Raw checkpoints, `run.json`, `metrics.jsonl`, `segments.jsonl`, snapshots, and
other large execution artifacts belong under the owning study/control's `results/generated/` directory and are ignored by Git.

## Tape model

All tape policies share the same identity-initialized learned writer and GQA readers at configurable decoder layers. Reader outputs are zero-initialized, and sequence-anchored RoPE is the default:


 ![Memory Access Through Cross-attention](/docs/memory_attn.png)

```yaml
variant: tape
memory_window: 32
memory_write_mode: dense       # dense | periodic | memory_token
memory_layers: [3, 7]          # or: all
memory_position_encoding: rope # default; explicit ablation: none
```

Periodic writes additionally require `memory_write_stride`. Explicit memory
slots use:

```yaml
variant: tape
memory_window: 32
memory_write_mode: memory_token
memory_write_stride: 8
memory_token_visibility: visible   # visible | write_only
```

`<MEM>` is an input-only architecture position with ID equal to the base vocabulary size `V`; it is not added to the LM output head. For physical input `A <MEM> B`, the language target at A is B, the MEM position has no LM loss, and `h_MEM` writes one tape record. See `docs/TAPE_MEMORY.md` for the exact attention, loss, cached-inference, and hybrid contracts.

Memory models can optionally continue NTP training with a causal next-memory prediction head. The head sees only `h_t`, targets detached final-pass future memory features, and is absent from inference. See `docs/NEXT_MEMORY_PREDICTION.md` for alignment, gradient, checkpoint, and leakage contracts.

## Training and cloud execution

The trainer supports exact resume on interruptible/spot instances: durable
checkpoint generations, newest-corrupt fallback, metrics repair, source/data
provenance, wall-clock and token checkpoint triggers, SIGINT/SIGTERM graceful
checkpointing, and weights-only scientific snapshots. See `docs/CLOUD.md`.

Memory-token runs distinguish linguistic data dose from physical transformer
work:

```text
unique_tokens_seen       = linguistic/data tokens
model_positions_seen     = ordinary + <MEM> physical positions
token_equivalent_compute = model positions x effective passes
```

Learning-rate schedules and run token budgets use linguistic tokens. Throughput
telemetry reports both linguistic tokens/s and model positions/s.

## Current research status

The active compute-conscious program is defined in
`benchmarks/development/experimental_pipeline.md`: local frozen-backbone
wiring, local Phase-B smoke tests, a resumable cloud pilot, selected
confirmation runs, and the locked six-arm Stage-5 continuation. It compares
vanilla, adaptive Recirculation, dense Tape, periodic-C32 Tape,
write-only explicit-`<MEM>`-C32 Tape, and their periodic-C32
Recirculation–Tape hybrid. Tape readers use layers `[3, 7]` except for the
hybrid's locked `[4, 7]` placement; no spacing, reader-placement, or controller
placement sweep is active.

FBT and MemoryAdd are retired controls. Their implementations and focused
correctness tests remain only for historical checkpoint/provenance
compatibility, like other archived research controls, but neither appears in
the active studies, efficiency qualifications, cloud campaign, or current
results. TapeAddHybrid, which uses the same MemoryAdd fast channel, is also
retired from the active experiment surface.

Historical benchmark results remain read-only evidence; they do not define the
active architecture API.

## Validate

```bash
uv sync --extra data --extra eval
make check
```

Without dependency installation, the source tree can also be tested in an
environment that already provides the locked dependencies with:

```bash
PYTHONPATH=src pytest -q
```

Prepare and verify the wiring and pilot artifacts with:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/verify_data.py data/dolmino/wiring_2048
uv run python scripts/prepare_data.py --config data/dolmino/pilot_2048/config.yaml
uv run python scripts/verify_data.py data/dolmino/pilot_2048
```

Before paid CUDA training, qualify batching using `benchmarks/efficiency/`, then
run the provider-agnostic preflight described in `docs/CLOUD.md`.
