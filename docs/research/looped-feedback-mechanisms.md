# Feedback mergers for preceding-token, parallel multi-pass training

Implementation update: the subsequent user decision moved both recurrent arms
to the attention variants' late emitted-memory source and selected only the
gated projected residual versus adaptive recirculation. The research below
records the earlier shortlist, not the active configuration. See
[the implemented contract](../RECURRENT_MEMORY.md).

Research checked 2026-09-02. Scope: possible merger operations for the existing pretrained TinyMistral, fixed-K, token-parallel experiment. Preserve the user's required route: preceding-token source from the previous pass into the current-token destination. Do not introduce same-token replacement or token-serial BPTT.

Recommendation: retain the current adaptive mixture as a baseline, but test a projected residual merger at the same layer pair. T²MLR is the closest architectural/training reference found; Huginn and Retrofitted Recurrence supply useful merger comparisons. None establishes a winner for this repository's frozen, low-K setting. This is a research recommendation, not an implementation or a change to the frozen benchmark contract.

Notation: `d[t,k]` is the current destination; `s[t-1,k-1]` is the permitted feedback source. In the same-position looped papers, `e[t]` is a fixed input/prelude representation and `h[t,k-1]` is the **same position's** previous-loop state. Replacing that operand with `s[t-1,k-1]` is our untested adaptation, not their published routing.

## Training: sequential computation is not sampled generation

The recirculation paper's adaptive training uses known text and BPTT (§D.5), not sampled continuations. Its exact replay schedule carries updated representations/KV state into later token steps (§2). Thus teacher forcing does not eliminate its temporal dependency. BPTT differentiates that dependency; it is not required because text is being sampled. The same section reports frozen-Gemma controller training, useful evidence for the merger under the paper's policy, not for our shifted policy. [Recirculation, §§2 and D.5](https://arxiv.org/html/2608.17981v2).

For our fixed-K computation, the feedback source for an entire pass is already available from the previous pass. Positions can therefore run in parallel, followed by differentiation through the chosen pass graph. Both schemes can be unrolled. What differs is the dependency graph and the extent of gradient propagation. Full BPTT is not the only possible estimator: truncation or an approximate parallel forward graph trades fidelity for cost.

## 1. T²MLR: closest match to the intended route

*T²MLR: Transformer with Temporal Middle-Layer Recurrence* (2026), §§2.2–2.4, injects a preceding-token middle-layer cache before an earlier layer. Equation (2.3) is:

```text
Phi(d,R) = d + tanh(gamma_cur) * sigmoid(f_cur([d,R])) ⊙ d
             + tanh(gamma_rec) * sigmoid(f_rec([d,R])) ⊙ W_rec R
```

The `f` maps are linear; both scalar `gamma`s start at zero. Its cache accumulates: `R_t = RMSNorm(h_end,t + R_(t-1))`. [Architecture](https://arxiv.org/html/2607.15178v1#S2).

Training uses parallel Jacobi refinement: normally 16 forward middle-block iterations and four backward iterations. Appendix B.4 compares against serial recurrence with **TBPTT-4**, not full-sequence BPTT. Section 4.4 reports pretrained-model retrofit gains; it is not a frozen-controller demonstration. [Training and retrofit evidence](https://arxiv.org/html/2607.15178v1).

Transfer limit: neither its accumulated cache nor its iteration budget matches our overwritten source and K=2/3 training. Borrowing its merger does not reproduce the complete method.

## 2. Full-bandwidth Transformer: relevant, already represented here

*Full-bandwidth Transformer* (2026), §3.1, uses `W_U h_(t-1) ⊙ sigmoid(W_G e_t)`: the projected previous latent supplies the value, while the current embedding gates it. It deliberately removes a direct additive embedding path. Section 3.3 trains shifted, token-parallel passes, applies losses across passes, and mixes occasional three-pass training into its curriculum. Its normalization and large-scale pretraining recipe matter to interpreting results. [Mechanism and training](https://arxiv.org/html/2608.08888v1#S3).

Our interpretation: this is a direct temporal-feedback precedent, but replacing the destination rather than preserving it is a more disruptive frozen-backbone retrofit. That is a reason to prioritize a residual candidate, not proof that FBT is inferior. The repo already has an FBT arm; this research does not justify reviving it in the active comparison without a separate decision.

## 3. Huginn: concatenation and learned projection

Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning* (2025), §§3.2–3.3: the recurrent core receives `A([h[t,k-1]; e[t]])`, with a learned `2d → d` linear map. Sandwich normalization stabilizes its transformer layers. Routing is same-position. The model is pretrained from scratch, with token-parallel passes and gradients through the last eight depth iterations; this is not token-serial BPTT. [Paper](https://arxiv.org/html/2502.05171v2#S3).

Evidence: §4.3 reports large-scale stability failures and improvements, but changes the merger, normalization, embedding scale, and learning rate together. That does not isolate the merger's causal contribution. The authors report addition working comparably at small scale. [Stability analysis](https://arxiv.org/html/2502.05171v2#S4.SS3).

The official `core_block_forward` confirms concatenation along the feature dimension, without a token shift. It also implements addition; code availability alone does not establish comparative effectiveness. [Author implementation](https://raw.githubusercontent.com/seal-rg/recurrent-pretraining/main/recpre/model_dynamic.py).

## 4. Retrofitted Recurrence: stronger evidence about addition versus projection

McLeish et al., *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence* (2025), §3: uses the same `2d → d` concatenation adapter, with same-position latent state plus persistent prelude output. It converts pretrained TinyLlama/Llama/OLMo layers and updates pretrained parameters, including the prelude; backward unrolling is truncated across depth, not tokens. [Architecture and training](https://arxiv.org/html/2511.07384v1#S3).

Appendix C.1.1/Figure 14 directly substitutes addition for the linear adapter: training loss is higher, while evaluation accuracy is approximately unchanged. This supports testing a simple merger; it does not establish a universal projection advantage. The broader results also depend on layer selection and a recurrence/healing curriculum. [Merger ablation and curriculum](https://arxiv.org/html/2511.07384v1).

Fit: a useful pretrained-model precedent for the *operation*, but not a tested shifted-feedback mechanism or frozen-backbone recipe.

## 5. Gated re-entry retrofit: concrete bridge, narrower task evidence

Shapiro, *Retrofitting Recurrent Depth into a Pretrained Language Model* (2026), §3.2/Listing 1: later loops use `u = h + g * (W_p N(p) + W_s h - h)`, omitting projection biases here. The learned scalar gate is identity-biased; the one-loop path bypasses the bridge. The operands share their token position. Each loop processes the whole sequence in parallel. [Architecture](https://arxiv.org/html/2608.11233v1#S3).

The paper tests both a trainable recurrent block and a frozen pretrained backbone with LoRA plus bridge. Evidence centers on controlled iterative chain tasks and supervised verbal transfer, not broad language-model quality or a clean ranking of merger alternatives. [Training budgets and evaluations](https://arxiv.org/html/2608.11233v1#S6).

Fit: illustrates normalization, split projections, gating, and an exact one-loop bypass; borrowing these for preceding-token routing remains an untested adaptation.

## 6. Ouro/LoopLM: its gate selects depth, not a merger

Zhu et al., *Scaling Latent Reasoning via Looped Language Models* (2025), §3.1/Equation (1): repeatedly applies the same transformer stack, passing each position's output directly into the next loop. There is no additional blend with a fresh early-layer representation in this definition. §3.2's sigmoid gate predicts **exit probability**, not how much feedback to mix. [Equations (1)–(3)](https://arxiv.org/html/2510.25741v4#S3).

Ouro is a large-scale pretraining system, including an upcycled larger branch, with token-parallel loop training and intermediate losses. Its results support the complete looped architecture and depth-allocation scheme, not a particular two-input feedback merger. [Training](https://arxiv.org/html/2510.25741v4#S4).

Fit: useful for loop objectives and evaluation, but direct state replacement conflicts with the requested route. Its exit gate should not be presented as a ready-made feedback gate.

## 7. Relaxed Recursive Transformers: adapters are in the weights

Bae et al., *Relaxed Recursive Transformers* (2024), §§2.2–2.3/Equations (2)–(4): loops shared transformer layers over the same-position state. Loop-specific LoRA modifies linear transformations as `Wx + BAx`; this is not a high-state/early-state merger. The pretrained model is converted through layer selection or averaging and SVD-based adapter initialization. All model parameters are subsequently trained. [Method](https://arxiv.org/html/2410.20672v3#S2).

Its ablations test initialization, LoRA rank, and relaxation of weight sharing; extended training/distillation recover performance. They do not rank additive, projected, or gated feedback mergers. [Ablations](https://arxiv.org/html/2410.20672v3#S3).

Fit: evidence for pretrained-model uptraining and pass-specific adaptation, not a drop-in merger. Adopting its weight-sharing/compression scheme would be a separate experiment.

## Implications for this repository — our interpretation

### What the current merger learns

The legacy [controller](../../src/tiny_mistral_mptt/variants/memory_modules.py) predicts independent, feature-wise alpha/beta vectors from a full-feature MLP. Its operation is `alpha ⊙ norm_match(s,d) + beta ⊙ d`. The gates can depend on all coordinates, but the transported source value has **no learned cross-feature projection**. Alpha and beta start at 0.1/0.9; their sum is not constrained to one after training. Later passes therefore perturb the pretrained representation at initialization.

A learned concatenation adapter is algebraically `W_d d + W_s s` (plus bias). Concatenation is not itself a special operation beyond those projections. The practical distinction worth testing here is whether the carried vector benefits from learned transport, and whether an explicit identity destination path helps preserve a pretrained backbone.

### Minimal candidate comparison

Keep `s = s[t-1,k-1]` and the existing source/destination layer pair in all candidates:

| Candidate | Operation | Purpose |
| --- | --- | --- |
| Existing adaptive mixture | `alpha(d,s) ⊙ norm_match(s,d) + beta(d,s) ⊙ d` | Preserve the established baseline. |
| Projected residual | `d + P N(s)` | Test a simple learned transport with an identity destination path. Initialize `P=0`. |
| Gated projected residual | `d + tanh(gamma) * sigmoid(G[d,N(s)] + b) ⊙ P N(s)` | Test conditional selection on top of transport. Initialize `gamma=0`, with nonzero `P`. |

Here `N` denotes a fixed, explicitly selected source-normalization rule shared by the two new candidates. The gated candidate is a simplification inspired by the cited work, **not** the complete T²MLR merger. Do not zero-initialize both its scalar gate and projection: that would initially block both learning paths. With zero scalar gating, the scalar learns first and the remaining branch follows.

If budget permits only one extra candidate, prioritize the gated projected residual. Include the simpler projected residual when possible so extra gating does not get credit for gains attributable to projection alone. This comparison still bundles changes in destination preservation and initialization relative to the existing mixture; a win would identify a useful design, not isolate one causal feature. Match those factors in follow-up ablations before claiming a mechanism-level explanation.

The existing [MemoryAdd](../../src/tiny_mistral_mptt/variants/memory_add.py) already implements zero-initialized projected residual feedback, but from the final layer into token embeddings. Reusing the idea at the recirculation layer pair would isolate placement more cleanly than directly comparing those two existing variants.

### Comparable experiments and low-sprawl implementation

Keep routing, source/destination layers, K distribution, final-pass objective, data, precision, and qualification budget fixed. Retain 2048-token blocks, the K=4 headline evaluation, K=1–4 diagnostics, and equal-budget per-model learning-rate sweeps. Report added parameters and actual training/inference cost; equal K is not equal cost across different architectures. Compare frozen and unfrozen outcomes separately, with the unfrozen run starting fresh as agreed.

Useful diagnostics are initial logit perturbation, feedback/destination norm ratio, gate saturation, and loss with real versus zeroed or mismatched feedback. Those checks help distinguish useful state transport from merely adding trainable capacity. They supplement, rather than replace, held-out NLL and downstream evaluations.

If implemented, select the merger through a small internal component at the existing feedback hook. Keep token shifting, layer routing, training, and evaluation shared. A different merger does not need a new inference policy, duplicated evaluator, or top-level experiment architecture. Rebuilding the backbone as a prelude/core/coda model, introducing an accumulated recurrent cache, or adding pass-specific adapters would be separate experiments.

The current code repeats the full backbone. For final-pass-only recirculation, its prefix before injection is unaffected by feedback and its tail after the source does not feed the next pass. That suggests a future prefix-cache/middle-repeat/final-tail execution optimization. It needs equivalence tests, especially for gradients and stochastic layers; it is not part of the merger recommendation or implemented here.

This note makes no claim that a sophisticated merger is superior. In particular, a system-level reasoning gain, an exit-gate ablation, and a state-merger ablation are different evidence.

Only this research note was edited for the research deliverable. No application code, benchmark manifests, training runs, or evaluation results were changed.
