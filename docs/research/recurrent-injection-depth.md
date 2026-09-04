# Recurrent memory at multiple injection layers

Research checked 2026-09-03. Scope: whether the existing late-emitted,
preceding-token recurrent memory should be read at more than one backbone layer.
This is an evidence note, not a change to the experimental protocol.

Conclusion: multiple read layers are a plausible ablation, particularly with a
frozen backbone, but the sources below do not establish that two reads outperform
one for this repository's architecture or low-K training objective. Keep number
of reads and their placement separate from the choice of merger.

## Primary-source evidence

### Feedback Transformer: a multi-layer feedback precedent

Fan et al. make a shared memory of past-token representations available to every
layer's attention. Their memory is a learned mixture of layer states, rather
than this repository's single late-emitted record. They report improvements on
language modeling and other sequential tasks. This demonstrates a workable
multi-layer feedback architecture; it is not a controlled one-versus-two-read
ablation, and it does not test frozen-backbone residual mergers or parallel
multi-pass training. [Architecture, section 3.2; experiments, section 4](https://arxiv.org/html/2002.09402v2#S3.SS2).

### P-Tuning v2: direct frozen-language-model evidence about placement

Liu et al. train prompts at multiple depths while freezing the language model.
Their BERT-large depth ablation compares equal numbers of prompted layers near
the input and near the output. Later-layer placements perform better in those
tests; on RTE, prompting layers 17–24 comes close to prompting all 24 layers.
This supports testing location rather than assuming every layer is needed.
The interventions are task-specific learned prefixes, not input-dependent
preceding-token state, and the tasks are NLU rather than next-token language
modeling. [Section 4.3 and Figure 3](https://aclanthology.org/2022.acl-short.8.pdf#page=5).

### Visual Prompt Tuning: more depths can help, but placement is task-dependent

Jia et al. freeze pretrained vision transformers and compare shallow prompting
with prompts introduced at multiple layers. Their depth ablation generally
improves with more prompted layers and finds early-layer prompts more useful
than later-only placement in that setup. This differs from the P-Tuning v2
placement result, cautioning against transferring a preferred depth mechanically
between models or tasks. Vision prompts are indirect adaptation evidence, not
recurrent-memory evidence. [Sections 3.2 and 4.3, Figure 7](https://arxiv.org/html/2203.12119v2#S4.SS3).

## Implications for this repository — hypotheses, not measured results

The [current recurrent contract](../RECURRENT_MEMORY.md) emits one normalized
top-state record through a shared writer. Each configured read layer receives
the same previous-token record and has its own merger. Reading that record at
layers `[3, 7]` would not create another memory, another previous-token hop, or
another backbone pass. It would create another opportunity to use the same
information within the pass.

A second read could restore useful feedback after intervening frozen layers
transform or attenuate the first injection. It could also give the later
representation a separately learned way to use that feedback. These are
motivations for a test, not evidence that the first injection is currently lost.

Costs and risks include another merger's parameters and computation, redundant
use of the same record, and stronger disruption of the pretrained computation.
The last risk matters for adaptive recirculation's nonzero initial mixture;
the zero-initialized projected residual instead preserves the original forward
computation at initialization. Extra reads do not by themselves prove greater
useful memory capacity.

Local constructor counts at hidden width 1024, including the shared writer but
excluding the backbone, make the capacity change explicit:

| Merger | Read layers `[3]` | Read layers `[3, 7]` |
| --- | ---: | ---: |
| Projected residual | 4,197,376 | 7,346,176 |
| Adaptive recirculation | 6,299,648 | 11,550,720 |

These are parameter counts from the [current implementation](../../src/tiny_mistral_mptt/variants/memory_modules.py),
not measured quality or latency results.

If pursued, `[3]`, `[7]`, and `[3, 7]` distinguish early placement, late placement,
and their combination. Comparing only `[3]` with `[3, 7]` cannot distinguish the
benefit of a later read from the benefit of multiple reads. Keep merger, writer,
data, K schedule, and evaluation fixed; give each configuration the same LR
qualification budget and report added parameters and measured cost. A gain with
two independent mergers would still bundle placement with trainable capacity.

No code, benchmark configuration, checkpoint, or scientific result was changed
for this note.
