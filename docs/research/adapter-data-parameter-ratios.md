# Data-to-trainable-parameter ratios in frozen-backbone adaptation

Checked 2026-09-05 against primary papers and first-party model metadata.

## Bottom line

The reviewed adapter and parameter-efficient fine-tuning papers do **not** use a
fixed data-to-trainable-parameter ratio as a stopping rule, adequacy threshold,
or quality criterion. They choose a fixed number of epochs or optimizer steps,
or select a checkpoint using held-out loss or task accuracy. The one paper that
jointly scales fine-tuning data and PEFT parameter count fits a task-dependent
power law and finds that increasing PEFT parameter count has little and
occasionally negative benefit; it explicitly says that the preferred
fine-tuning method and crossover data size do not generalize across tasks.

Consequently, a ratio such as “20 tokens per trainable parameter” is not a
published adapter-training rule. It is especially unsafe to transfer the
pretraining heuristic to a frozen-backbone retrofit: the frozen backbone still
provides almost all of the representation and computation, while the added
parameters control a restricted intervention into that representation.

## Definitions and comparability

Two ratios are easy to conflate:

- **Unique-data ratio:** distinct linguistic training tokens divided by
  trainable parameters.
- **Presentation ratio:** all token positions processed across optimizer steps,
  including repeated epochs, divided by trainable parameters.

Most PEFT papers report examples, epochs, maximum sequence lengths, or steps,
not tokenizer-specific counts of actual linguistic tokens. Their ratios are
therefore often recoverable only as examples per parameter, a padded-token
upper bound, or a presentation estimate. These are not interchangeable with
this repository's exact counter of linguistic tokens.

## Primary precedents

| Source and setting | Trainable parameters | Reported data or optimization budget | Recoverable ratio | Selection or stopping rule |
| --- | ---: | --- | --- | --- |
| [Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*](https://arxiv.org/html/1902.00751), frozen BERT with bottleneck adapters | GLUE uses 3.6% of BERT-Large's 330M parameters, about 11.9M. The additional-task suite reports 1.14% of BERT-Base per task, about 1.25M using BERT-Base's 110M count. | The 17 additional tasks span about 900 to 330K training examples. Adapter schedules are selected from 20, 50, or 100 epochs by inspecting validation curves; Table 3 gives exact example counts and Table 4 the chosen epochs. | No linguistic-token ratio is reported. Even within one study, dataset size spans more than two orders of magnitude while the same adapter family remains viable. | Best validation accuracy; epoch count chosen from validation learning curves, not from examples or tokens per adapter parameter. [Methods and tables](https://arxiv.org/html/1902.00751#S3), [BERT parameter counts](https://arxiv.org/abs/1810.04805). |
| [Hu et al., *LoRA*](https://arxiv.org/html/2106.09685), frozen GPT-3 175B | The main GPT-3 LoRA row uses 4.7M trainable parameters; ablations span 4.7M to 603.8M. | WikiSQL has 56,355 training examples, MNLI has 392K, and SAMSum has 14,732. GPT-3 runs use two epochs and maximum sequence lengths 384, 768, and 2,048 respectively. | For the 4.7M row, unique examples per parameter are about 0.012 (WikiSQL), 0.083 (MNLI), and 0.0031 (SAMSum). Two-epoch example-presentation ratios are twice those values. Actual linguistic-token counts are not reported. The maximum padded-position bounds would be 9.2, 128.1, and 12.8 positions per trainable parameter, showing why max length is not a reliable token count. | Fixed two-epoch schedule with task-specific learning-rate tuning. The paper compares validation accuracy/ROUGE and shows that performance stabilizes or can worsen as trainable parameter count rises; it does not stop at a data/parameter ratio. [GPT-3 results and budgets](https://arxiv.org/html/2106.09685#S5.SS5), [dataset details](https://arxiv.org/html/2106.09685#A3), [parameter ablation](https://arxiv.org/html/2106.09685#A6.SS2). |
| [Liu et al., *Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper than In-Context Learning*](https://arxiv.org/html/2205.05638), T-Few/(IA)3 on a frozen T0 backbone | 540K for (IA)3 on T0-3B in the PEFT comparison. | Downstream training is 1,000 steps × 8 sequences. The paper's FLOP estimate uses a median 103 tokens per training sequence, or about 824K presented tokens. The median task has 41 distinct examples. The (IA)3 parameters are first pretrained for 100K steps × 16 sequences on T0's multitask mixture. | The downstream presentation estimate is about 1.53 tokens per trainable parameter; the median unique set is only about 4,223 tokens, or 0.0078 unique tokens per parameter. Both figures omit or separate the substantial multitask pre-adaptation of (IA)3. | A fixed 1,000-step recipe, deliberately shared across tasks without per-task tuning. Performance is measured at the end of training. No ratio criterion is used. [Recipe and pre-adaptation](https://arxiv.org/html/2205.05638#S3.SS4), [cost calculation](https://arxiv.org/html/2205.05638#S4.SS2), [parameter table](https://arxiv.org/html/2205.05638#A4). |
| [Zhang et al., *LLaMA-Adapter*](https://arxiv.org/html/2303.16199), frozen LLaMA-7B | 1.2M for LLaMA-Adapter; the same table reports 4.2M for Alpaca-LoRA. | Both use 52K instruction examples. LLaMA-Adapter trains for five epochs with batch size 64, giving 260K example presentations; token counts and sequence-length distribution are not reported. | Unique-example ratios are about 0.043 for the 1.2M adapter and 0.012 for the 4.2M LoRA; presentation ratios are about 0.217 and 0.062. | Fixed five epochs, followed by task and GPT-4-based response evaluation. No data/parameter threshold. [Training setup and efficiency table](https://arxiv.org/html/2303.16199#S4.SS1). |
| [Zhang et al., *When Scaling Meets LLM Finetuning*](https://arxiv.org/html/2402.17193), prompt tuning and LoRA on frozen 1B–16B LMs | On the 1B model, the default prompt has exactly 100 × 2,048 = 204,800 parameters. The paper says default rank-4 LoRA is about 0.19% of the roughly 1B backbone (about 1.9M); ranks 4–128 scale this count linearly to about 60.8M. | PEFT data sweeps use 8K–100K examples. Runs are capped at 200K steps or 100 epochs, whichever comes first; PEFT batch size is 16. | The experiment deliberately crosses data size with PEFT parameter size rather than holding a ratio fixed. It finds PEFT-parameter scaling exponents with absolute value below 0.01 and sometimes inverse scaling. Fine-tuning-data effects are larger but task dependent. | Best checkpoint by development token-level perplexity within the step/epoch cap. The paper's fitted law is a quality model, but it is a multiplicative function of data and parameters with fitted task-specific exponents—not a constant data/parameter rule. [Setup](https://arxiv.org/html/2402.17193#S2), [scaling results](https://arxiv.org/html/2402.17193#S4), [optimization](https://arxiv.org/html/2402.17193#A1). |
| [Mozer et al., *Recirculation*](https://arxiv.org/html/2608.17981v2#A4.SS5), adaptive frozen-backbone residual mixing | The conditional-vector controller is a two-hidden-layer MLP. With Gemma-3 1B's official hidden size `d=1152`, its `2d -> d -> d -> 2d` matrices contain `5d^2 = 6,635,520` weights, approximately 6.64M trainable parameters after small bias/normalization terms. This is a derived count because the paper does not print the total. | 100 steps × batch 32 × full 1,024-token windows = 3,276,800 token presentations. Training draws from 250 documents each from arXiv, C4, and PG19; the paper does not state the distinct-token count of the resulting full-window pool. | About 0.49 presented tokens per derived trainable parameter. This is the closest reviewed architectural precedent to the repository's learned frozen-backbone feedback wiring. | Fixed 100 steps. Quality is held-out perplexity reduction across nine datasets and downstream accuracy; the ratio is not used to stop or qualify the run. [Adaptive method](https://arxiv.org/html/2608.17981v2#S4.SS6), [training details](https://arxiv.org/html/2608.17981v2#A4.SS5), [official Gemma-3 1B configuration](https://huggingface.co/google/gemma-3-1b-pt/blob/main/config.json). |

### The closest explicit use of ratio language

[Farina et al., *Rethinking Few-Shot Adaptation of Vision-Language Models in
Two Stages*](https://openaccess.thecvf.com/content/CVPR2025/html/Farina_Rethinking_Few-Shot_Adaptation_of_Vision-Language_Models_in_Two_Stages_CVPR_2025_paper.html)
is the closest reviewed paper to treating the ratio as an explanatory variable.
On frozen CLIP ViT-B/16 it studies 16-shot tuning with a fixed 8,000-step
budget. Its supplement reports 61K LayerNorm, 184K LoRA, and 125K BitFit
parameters, and *speculates* that LayerNorm's more balanced data-to-parameter
ratio on datasets with more classes has a regularizing effect. The authors do
not quantify a target ratio, validate a universal threshold, or stop training
when a ratio is reached. Their proposed method instead fixes a maximum step
budget and tunes how the budget is divided between two stages; its default
budget is `300 × shots` steps. See the [accepted-paper supplement, Sections D–E](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Farina_Rethinking_Few-Shot_Adaptation_CVPR_2025_supplemental.pdf).

### What the pretraining ratio does and does not say

[Hoffmann et al., *Training Compute-Optimal Large Language Models*](https://arxiv.org/html/2203.15556#S3)
fit a compute-optimal frontier for models trained from scratch. Their Table 3
projects 20.2B training tokens for a 1B-parameter model, about 20.2 tokens per
parameter, and Chinchilla itself used 1.4T tokens for 70B parameters, exactly 20
tokens per parameter. In that experiment, the denominator is the full model
being learned and the criterion is minimum pretraining loss under a fixed FLOP
budget. It does not test a pretrained frozen backbone, added adapters, or the
effective capacity of an adapter/backbone pair. The numerical resemblance to an
adapter ratio is therefore coincidental, not a transferable stopping rule.

## Implications for this repository

The active frozen comparison contains 4,195,000–4,197,376 added parameters in
the one-site group and 7,341,424–7,346,176 in the two-site group. Its exact
planned budget is 100,007,936 unique linguistic tokens; its completed learning-
rate qualification used 5,013,504 linguistic token presentations per run. See
the repository's [frozen comparison protocol](../../benchmarks/development/frozen_backbone_comparison/README.md)
and [qualification protocol](../../benchmarks/development/frozen_backbone_lr_qualification/README.md).

Using the rounded 4.2M and 7.3M budgets requested for planning:

| Stage | One site, 4.2M | Two sites, 7.3M |
| --- | ---: | ---: |
| 5.0135M-token LR qualification | 1.19 tokens/trainable parameter | 0.69 tokens/trainable parameter |
| 100.008M-token frozen comparison | 23.81 tokens/trainable parameter | 13.70 tokens/trainable parameter |

Practical conclusions:

1. **The 5M qualification is defensible as a rate/stability screen, not as
   convergence evidence.** Its ratios are already above adaptive
   Recirculation's approximately 0.49 and near T-Few's approximately 1.53
   downstream presentation estimate. Those precedents show that a ratio below
   one need not make a frozen controller untrainable, but their tasks,
   initialization, repetition, and losses differ too much to predict quality
   here.
2. **The 100M budget is substantial by direct frozen-adaptation precedent, but
   “crossing 20” proves nothing.** It should provide a meaningful learning curve
   for both groups. Promotion or stopping should depend on held-out NLL level and
   slope, real/zero/mismatched-memory controls, downstream feedback evaluation,
   and run stability—not on reaching 20 tokens per added parameter.
3. **Equal token budgets do not equal equal per-parameter exposure across site
   groups.** The two-site ratio is about 42.5% lower than the one-site ratio.
   That reinforces the protocol's decision to compare architectures within each
   parameter-matched site group and to treat one-site versus two-site as a
   practical architecture ablation rather than a pure injection-count result.
4. **Record both unique and presented tokens.** The active 100M artifact is
   intended as unique data, whereas multi-epoch PEFT precedents often reuse tiny
   datasets. Reporting optimizer steps, token presentations, and distinct corpus
   tokens separately prevents an apparently comparable ratio from hiding very
   different data reuse.
5. **If a budget extension is considered, use an empirical continuation gate.**
   Extend only when held-out NLL is still improving materially near 100M and the
   feedback-specific controls remain healthy. A small pilot extension can
   estimate marginal gain per additional token; the reviewed literature gives
   no authoritative ratio at which these adapters should stop.

## Claim limit

This review found no ratio-based stopping or quality rule in the selected
primary precedents. It is not proof that no such proposal exists anywhere. The
strongest directly relevant evidence is empirical: frozen-backbone adaptation
can work over ratios far below and above one, while quality and the benefit of
larger PEFT modules remain task-, backbone-, architecture-, and optimization-
dependent.
