# Attention vs. Recurrence in Multi-Pass Transformers

**tl;dr:** Recent work such as the [full bandwidth transformer](https://arxiv.org/abs/2608.08888) and [recirculation](https://arxiv.org/abs/2608.17981) suggests that recurrently feeding intermediate representations back through a transformer gives you performance gains almost for free. I found that replacing the recurrent mechanisms with attention over a bank of previous-pass states performs better, including when that attention is sparse (and thus not always attending to recent hidden states).

The idea behind [multi-pass training](https://github.com/PeterBjerreHansen/multi-pass-transformer-training) goes something like this: Pass 1 is an ordinary transformer pass. Pass 2 runs the same transformer again, but can use additional hidden states produced on pass 1. The question is if that information should arrive through a recurrent connection or through attention over a bank of previous-pass states. To test this I first retrofitted [a TinyMistral model](https://huggingface.co/M4-ai/TinyMistral-248M-v3) into doing feedback inference by training it with multi-pass Jacobi-style updates. The new parameters were first wired into the frozen backbone for 5 million tokens, then the whole models were trained on 100M tokens from the [OLMo2 annealing mixture](https://huggingface.co/datasets/allenai/dolmino-mix-1124).

| Model / method                                | Final validation PPL | Late PPL reduction, 50M → 100M | Total parameters | Relative training FLOPs |
| --------------------------------------------- | -------------------: | -----------------------------: | ---------------: | ----------------------: |
| Transformer baseline                          |                7.778 |                  0.110 (1.40%) |         248.024M |                  1.000x |
| Sparse SWA control (not yet run)              |                    - |                              - |         248.024M |                       - |
| Adaptive Recirculation                        |                7.678 |                  0.110 (1.42%) |         253.275M |                  2.127x |
| Sparse Memory-token Bank                      |                7.616 |                  0.133 (1.72%) |         254.321M |                  2.187x |
| Sparse Periodic Bank                          |                7.599 |                  0.113 (1.46%) |         254.320M |                  2.122x |
| Dense Bank                                    |                7.534 |                  0.112 (1.46%) |         254.320M |                  2.133x |
| Adaptive Recirculation + Sparse Periodic Bank |                7.519 |                  0.120 (1.57%) |         259.571M |                  2.149x |
| Multiscale Bank control (not yet run)         |                    - |                              - |         254.320M |                       - |

I think there are three notable patterns in these results:

1. The attention-based mechanisms outperform the recurrent mechanisms almost uniformly. Since replacing recurrent connections with attention has a solid track record, I'd wager this trick would work at scale as well.

2. The sparse attention to far-away memories outperformed models with only the recurrent connections to memories. To me this result slightly favours the notion that the performance gains reported in the FBT and recirculation papers stem from *mere greater effective depth* rather than *unlocking recurrent computation patterns*.

3. The recurrent/attention hybrid performs on par with the dense attention model, and so attention over a sparse memory bank could provide a valuable "slow" (long-range) memory for multi-pass models with only the "fast" (short-term) recurrent pattern.

The experiment is riddled with confounders such as differing compute budgets, parameter count, and starting-point inequality (some mechanisms are more easily slotted into a pretrained model, and the vanilla backbone had no pre-run adaptation). Moreover, I simply do not have the computational resources to provide the exhaustive sweeps and ablations one needs to make convincing optimality arguments. However, I do think the results point towards potential improvements to the existing architectures.

### The memory-bank models
In recurrent patterns, like that those of FBT and adaptive Recirculation, token $t$ on pass $k$ receives essentially one shifted state, $m_{t-1}^{k-1}$.
> **Recurrent feedback:**<br>
> $`h_t^{(k)} = \mathrm{Mix}\left(h_t^{(k)}, m_{t-1}^{(k-1)}\right)`$

The bank models replace the narrow recurrent feedback path with cross-attention over a bounded set of states from the preceding pass. A bank reader instead chooses among up to $W$ strictly earlier memory records:
> $`\textbf{Memory Bank:}`$<br>
> &nbsp;&nbsp; $`B_t^{(k)} = \left[\mathrm{Writer}\left(h_s^{(k-1)}\right) \;\middle|\; s<t,\ \mathrm{write}(s)\right]_{\text{ W most recent} }`$<br>
> &nbsp;&nbsp; $`h_t^{(k)} = h_t^{(k)} + \mathrm{CrossAttention}\left(Q=h_t^{(k)},\ KV=B_t^{(k)}\right)`$

where the writer is some policy for selecting what subset of memories should be in the bank. Our writer also applies a linear transformation to the selected latent states.

![Memory Access Through Cross-attention](/docs/memory_attn.png)

Recirculation fixes the feedback source in advance, while Bank attention lets each token content-select which previous-pass representations are useful. The bank readers are separate GQA cross-attention residuals inserted at selected decoder layers; memory positions retain their original sequence coordinates for RoPE. The original Bank variants differ in **what gets written**. Multiscale Bank instead changes which dense writes the reader retains:

| Variant               | Storage policy             | Attention type                     |
| --------------------- | -------------------------- | ---------------------------------- |
| **Dense Bank**        | every token                | local dense SWA                    |
| **Periodic Bank**     | every $s$-th token         | long-range sparse                  |
| **Memory-token Bank** | explicit `<MEM>` positions | long-range sparse on memory states |
| **Multiscale Bank**   | every token                | dense SWA plus sparse older        |

`memory_window: 32` means **32 retained memory records**, not 32 tokens. Thus a sparse periodic bank with stride 32 can expose a token to memories spread over roughly $32\times$ the sequence range of a dense bank with the same capacity.

The hybrid combines both routes: adaptive Recirculation provides a **fast**, local feedback channel, while the sparse bank provides a **slow**, longer-range content-addressed memory. This is the motivation for viewing the two mechanisms as complementary rather than mutually exclusive. Multiscale Bank is the attention-only control for the recurrent/Bank hybrid. It gives each Bank reader both dense recent records and sparse older records in one softmax, but removes the recurrent channel.

Sparse SWA is the vanilla Transformer control. At selected layers, it extends ordinary sliding-window self-attention with sparse attention to fixed past tokens. It is one-pass, reads current-pass token states rather than a previous-pass Bank, and adds no parameters.

```yaml
variant: bank

memory_window: 32

memory_write_mode: dense       # dense | periodic | memory_token

memory_layers: [3, 7]

memory_position_encoding: rope
```

Periodic and memory-token banks additionally set `memory_write_stride`. In memory-token mode, `<MEM>` is input-only: it is not part of the LM output vocabulary and receives no direct LM loss. In the `write_only` configuration used in the main comparison, later tokens can access its state only through the bank. See `docs/BANK_MEMORY.md` for the exact attention masks, write timing, cached-inference behavior, and hybrid contracts. See `docs/ARCHITECTURES.md` for the two attention-control architectures.

### A Note on Efficiency

`Relative training FLOPs` is normalized to the vanilla K=1 run at 1.000x. It uses the 90% K=2 / 10% K=3 training schedule, so most of the $>2x$ multiplier is because I chose to always run at least two passes. This policy puts more pressure on the memory adaptations (which is what I wanted to study most), but is presumably computationally suboptimal since single passes will adapt the backbone to the training distribution more efficiently. The measure counts mostly the dominant dense matrix products in the forward and backward passes, and you can see the estimator in `scripts/estimate_training_flops.py`.

### More Ideas, Future Work

1. **Next Memory Prediction.** This branch implements a training-only auxiliary objective inspired by [NextLat](https://arxiv.org/abs/2511.05963), with future memory representations as targets. I have a strong and unproven suspicion that this will work nicely: since there is a lot of pressure on the memory latents to be useful *inputs* to the model, they should contain information that is already useful for NTP prediction and so should be less prone to collapse. In some sense, predicting these memories amounts to predicting features that the model will think are useful later. I like this notion that one should want to predict what is itself predictive. The implementation is tested, but I have not yet evaluated whether NMP improves language-model quality. See `docs/NEXT_MEMORY_PREDICTION.md`.

2. The sparse SWA and Multiscale Bank controls are implemented. Their pending runs will test whether the hybrid needs a recurrent path.

3. Most SOTA models still use dense attention in at least a few layers. I suspect that a local SWA attention + dense long-range attention over memories might perform better than placing the dense attention in a normal layer.

4. I want to properly wire in the [FBT](https://arxiv.org/abs/2608.08888) model next and test out hybrids. However, the process is a little more tricky for FBT as the mechanism is not residual.

## Repository map

* `src/tiny_mistral/`: validated vendored vanilla TinyMistral implementation.
* `src/tiny_mistral_mptt/`: research architectures, training, evaluation, and inference.
* `benchmarks/`: controls, development/core studies, and engineering efficiency measurements.
* `data/`: deterministic dataset recipes; generated artifacts are local/ignored.
* `evaluation/`: reusable evaluation-suite definitions.
* `docs/`: architecture, data, training, inference, cloud, and validation contracts.

There is intentionally no central `configs/` directory. Runnable settings live with the study or asset that owns them. Development/core studies use `STUDY.yaml` for the scientific question and comparison structure; runnable YAML files remain the execution source of truth.

Raw checkpoints, `run.json`, `metrics.jsonl`, `segments.jsonl`, snapshots, and other large execution artifacts belong under the owning study/control's `results/<arm>/` directory and are ignored by Git.

## Training and cloud execution

The trainer supports exact resume on interruptible/spot instances: durable checkpoint generations, newest-corrupt fallback, metrics repair, source/data provenance, wall-clock and token checkpoint triggers, SIGINT/SIGTERM graceful checkpointing, and weights-only scientific snapshots. See `docs/CLOUD.md`.

Memory-token runs distinguish linguistic data dose from physical transformer work:

```text
unique_tokens_seen       = linguistic/data tokens

model_positions_seen     = ordinary + <MEM> physical positions

token_equivalent_compute = model positions x effective passes
```

Learning-rate schedules and run token budgets use linguistic tokens. Throughput telemetry reports both linguistic tokens/s and model positions/s.

## Current research status

The locked eight-arm study compares vanilla, adaptive Recirculation, three Bank write policies, the Recirculation–Periodic Bank hybrid, Multiscale Bank, and Sparse SWA. Its configs remain under `benchmarks/core/stage_5_cloud_100m/`; the six original arms are complete and the two attention controls are the remaining runs. Multiscale Bank first needs frozen-backbone wiring. Sparse SWA has no added parameters and starts in Phase B. The runnable path is documented in `benchmarks/development/experimental_pipeline.md`.

Next Memory Prediction is implemented and tested on this branch, but does not yet have a reported language-model quality result.

FBT and MemoryAdd are retired controls. Their implementations and focused correctness tests remain only for historical checkpoint/provenance compatibility, like other archived research controls, but neither appears in the active studies, efficiency qualifications, cloud campaign, or current results. BankAddHybrid, which uses the same MemoryAdd fast channel, is also retired from the active experiment surface.

Historical benchmark results remain read-only evidence; they do not define the active architecture API.

## Validate

The repository supports Python 3.10–3.13 and uses `uv` for its reproducible environment.

```bash
uv sync --extra data --extra eval

make check
```

Without dependency installation, the source tree can also be tested in an environment that already provides the locked dependencies with:

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

Before paid CUDA training, qualify batching using `benchmarks/efficiency/`, then run the provider-agnostic preflight described in `docs/CLOUD.md`.
