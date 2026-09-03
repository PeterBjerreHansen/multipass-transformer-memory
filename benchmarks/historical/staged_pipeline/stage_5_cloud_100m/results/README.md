# Stage 5 results

The compact eight-arm result table and original commentary are below. All eight
arms are complete. The retained artifacts belong directly under ignored
`<arm>/` directories and contain metadata, metrics, logs, and selected
evaluations. Large checkpoints were intentionally not transferred, and only
the explicitly verified transfers have a checksum manifest.

The retained `evaluation_full_mps.json` files were produced before checkpoint,
config, suite, seed, and sample-level evidence became mandatory. Several are
numerically identical and cannot be audited from the files alone. The legacy
adapter also inherited fixed 512-token padding/truncation from `tokenizer.json`,
which could reduce lm-eval candidate continuations to empty encodings and return
zero log-likelihoods. Treat these files as invalid capability evidence. Rerun
the milestone snapshots with `scripts/evaluate_lm_harness.py`; do not overwrite
the historical files.

## Archived README discussion

The following discussion was moved from the root README during the documentation audit.
It describes the historical screen, not the active frozen study. The FBT investigation
mentioned below is no longer retained in the archive. Result values and the
original interpretation are preserved.

## Historical architecture screen

**tl;dr:** Recent work such as the [full bandwidth transformer](https://arxiv.org/abs/2608.08888) and [recirculation](https://arxiv.org/abs/2608.17981) suggests that recurrently feeding intermediate representations back through a transformer can improve performance. In this exploratory, equal-token experiment, attention over previous-pass states produced lower same-stream validation NLL than the recurrent controls. The result is not an equal-compute comparison and does not yet establish downstream capability gains.

The results below are the completed historical architecture screen. Its exact
protocol and artifacts are preserved under
`benchmarks/historical/staged_pipeline/`; they do not define the current paper
experiment contract.
The recirculation entries describe this repository's preceding-token retrofit,
not a reproduction of the paper's removed same-token replay/training policy.

The active frozen comparison now uses a shared late-memory writer for recurrent
and attention variants. Its two recurrent arms compare an identity-initialized
gated projected residual with adaptive recirculation mixing, both reading only
the preceding token's emitted memory. Dense, strided and dense-and-strided Memory
Attention complete the five-arm comparison. See
[the recurrent-memory contract](../../../../../docs/RECURRENT_MEMORY.md); the historical results
below do not measure these restructured recurrent arms.

The idea behind [multi-pass training](https://github.com/PeterBjerreHansen/multi-pass-transformer-training) goes something like this: Pass 1 is an ordinary transformer pass. Pass 2 runs the same transformer again, but can use additional hidden states produced on pass 1. The question is if that information should arrive through a recurrent connection or through cross-pass attention over previous-pass states. To test this I first retrofitted [a TinyMistral model](https://huggingface.co/M4-ai/TinyMistral-248M-v3) into doing feedback inference by training it with multi-pass Jacobi-style updates. The new parameters were first wired into the frozen backbone for 5 million tokens, then the whole models were trained on 100M tokens from the [OLMo2 annealing mixture](https://huggingface.co/datasets/allenai/dolmino-mix-1124).

| Model / method                                | Final monitoring PPL | Late PPL reduction, 50M → 100M | Total parameters | Relative training FLOPs |
| --------------------------------------------- | -------------------: | -----------------------------: | ---------------: | ----------------------: |
| SWA Transformer baseline                          |                7.778 |                  0.110 (1.40%) |         248.024M |                 1.0000x |
| Strided Attention control                            |                7.811 |                  0.121 (1.52%) |         248.024M |                 1.0004x |
| 🔴 Full Bandwidth Transformer| - | - | - | - |
| Adaptive Recirculation                        |                7.678 |                  0.110 (1.42%) |         253.275M |                 2.1267x |
| Strided Memory Attention               |                7.599 |                  0.113 (1.46%) |         254.320M |                 2.1222x |
| Dense SWA Memory Attention                        |                7.534 |                  0.112 (1.46%) |         254.320M |                 2.1327x |
| Adaptive Recirculation + Strided Memory Attention |              7.519 |                  0.120 (1.57%) |         259.571M |                 2.1489x |
| Dense SWA + Strided Memory Attention           |                7.504 |                  0.111 (1.46%) |         254.320M |                 2.1332x |

I think there are three notable patterns in these results:

1. Within this protocol, the attention-based mechanisms have lower monitoring PPL than the recurrent mechanisms almost uniformly. Whether that transfers to other scales or downstream generation remains an open question.

2. The sparse attention to far-away memories outperformed models with only the recurrent connections to memories. To me this result slightly favours the notion that the performance gains reported in the FBT and recirculation papers stem from **mere greater effective depth** rather than **unlocking recurrent computation patterns**.

3. The recurrent/attention hybrid performs on par with the short-range dense attention only model, and so Strided Memory Attention could provide a valuable "slow" (long-range) memory for multi-pass models with only the "fast" (short-term) recurrent pattern. However, supplying the short-range dense attention only model with strided long-range attention improves performance even further at a smaller compute budget, and so the attention only variant looks superior.

The experiment has important confounders: differing compute budgets, parameter counts, and starting points (some mechanisms are more easily slotted into a pretrained model, and the SWA Transformer backbone had no pre-run adaptation). The staged validation artifact also overlaps earlier wiring training data, so its PPL is a convergence and same-stream diagnostic rather than independent held-out evidence. Exhaustive sweeps and ablations would be required for optimality claims. The present results should therefore be read as architecture-screening evidence and hypotheses for independently evaluated follow-up work.

> 🔴 **FBT is omitted from the common-protocol comparison for now since it requires some architecture-specific initialization and pretraining adjustments. It performs poorly when wired and/or trained with the common protocol, but it might do better in a comparison with architecture-specific optimisation.**

### The Memory Attention models

In recurrent patterns, like those of FBT and adaptive Recirculation, token $t$ on pass $k$ receives essentially one predetermined shifted state, $m_{t-1}^{k-1}$:

> **Recurrent feedback:**<br>
> $`h_t^{(k)} = \mathrm{Mix}\left(h_t^{(k)}, m_{t-1}^{(k-1)}\right)`$

Memory Attention instead exposes previous-pass states as a separately addressable key/value source. Token $t$ may attend to some causally valid subset of the previous-pass memory state, which I denote by the access pattern $A_t$:

> **Cross-pass memory attention:**<br>
>    $`h_t^{(k)} = h_t^{(k)} + \mathrm{CrossAttention}\left(Q=h_t^{(k)},\ KV=M^{(k-1)};\ \mathrm{mask}=A_t\right)`$

Here $M^{(k-1)}$ denotes the previous-pass memory states and $A_t$ determines which of those states token $t$ is allowed to access.

![Memory Access Through Cross-attention](../../../../../docs/memory_attn.png)

Recirculation fixes the feedback source in advance, while Memory Attention lets each token content-select among the previous-pass representations made available to it. Memory-attention readers are separate GQA cross-attention residuals inserted at selected decoder layers; memory positions retain their original sequence coordinates for RoPE.

The Memory Attention variants differ primarily in their **memory access pattern**:

| Variant               | Accessible memories                      | Attention pattern |
| --------------------- | ---------------------------------------- | ----------------- |
| **Dense Memory Attention**        | recent previous-pass states              | local dense SWA   |
| **Strided Memory Attention**      | regularly strided previous-pass states   | long-range sparse |
| **Memory-token Attention** | explicit `<MEM>` states                  | long-range sparse |
| **Dense-and-strided Memory Attention**   | dense recent + sparse older states       | dense + strided   |

These access patterns admit several equivalent conceptual realizations. For example, sparse Memory Attention can be understood either as retaining only the memory states that will be addressable, or as retaining a denser memory and masking the inaccessible states during attention. The implementation uses selective memory writes and bounded retained KV records for efficiency, but that is not essential to the Memory Attention abstraction itself.

`memory_window: 32` therefore refers to the implementation's capacity of **32 retained addressable memory records**, not a 32-token receptive field. For a Strided Memory Attention model with stride 32, those 32 records can represent memories spread over roughly $32\times$ the sequence range of a dense model with the same retained capacity. The hybrid combines both routes: adaptive Recirculation provides a **fast**, local feedback channel, while Strided Memory Attention provides a **slow**, longer-range content-addressed memory. This is the motivation for viewing the two mechanisms as complementary rather than mutually exclusive. Dense-and-strided Memory Attention is the attention-only control for the recurrent/Memory Attention hybrid. It gives each memory-attention reader access to both dense recent memories and sparse older memories in one softmax, but removes the recurrent channel. Strided Attention is the SWA Transformer control. At selected layers, it extends ordinary sliding-window self-attention with strided attention to fixed-stride past tokens. It is one-pass, reads current-pass token states rather than previous-pass memory records, and adds no parameters.

The current implementation configures these access patterns through the memory-write policy:

```yaml
variant: memory_attention

memory_window: 32

memory_write_mode: dense       # dense | strided | memory_token

memory_layers: [3, 7]

memory_position_encoding: rope
```

Strided and memory-token Memory Attention additionally set `memory_write_stride`. In memory-token mode, `<MEM>` is input-only: it is not part of the LM output vocabulary and receives no direct LM loss. With `write_only`, later tokens can access its state only through Memory Attention. Memory-token attention is supported but is not one of the active five arms.

See `docs/MEMORY_ATTENTION.md` for the conceptual vocabulary, implementation-level
attention masks, retention/write timing, cached-inference behavior, and hybrid
contracts. See `docs/ARCHITECTURES.md` for the two attention-control architectures.

### A Note on Efficiency

`Relative training FLOPs` is normalized to the SWA Transformer K=1 run at 1.000x. It uses the 90% K=2 / 10% K=3 training schedule, so most of the $>2x$ multiplier is because I chose to always run at least two passes. This policy puts more pressure on the memory adaptations (which is what I wanted to study most), but is presumably computationally suboptimal since single passes will adapt the backbone to the training distribution more efficiently. The measure counts mostly the dominant dense matrix products in the forward and backward passes, and you can see the estimator in `scripts/estimate_training_flops.py`.

### Follow-up ideas from the historical screen

1. Most SOTA models still use dense attention in at least a few layers. I suspect that a local SWA attention + dense long-range attention over memories might perform better than placing the dense attention in a normal layer.

2. A later FBT wiring investigation was attempted but is not retained in this
   checkout. It is not part of the current comparison because its initialization
   and pretraining requirements differ.
