# Frozen-Wiring Experiments: Grill Exchange

> Historical transcript. It includes superseded proposals and removed methods.
> Use the [current documentation map](README.md) and [development plan](DEVELOPMENT_PLAN.md)
> for the implemented contract and the latest user decisions.

The historical data settings discussed in this transcript were retired during
the clean-slate reset on 2026-09-04. The active data namespace now contains
only `data/dolmino/wiring_2048` and `data/dolmino/gpu_2048`; references below to
pilot, staged, long-run, or Stage-6 recipes are preserved as conversation
context, not as preparation instructions.

This transcript contains the substantive exchange about the frozen-wiring experiments. Internal tool-call payloads and environment metadata are omitted.

## Opening request

### User

Please grill me about the frozen-wiring experiments both conceptually and execution.

### Assistant

I’m using the grilling skill to turn this into a decision tree: first I’ll map what the frozen-wiring experiments actually are from the repository, then I’ll challenge the highest-leverage conceptual and execution assumptions in rounds. I’ll wait for your answers between rounds.

## Round 1 — Define the claim and campaign

The repository currently distinguishes the planned four-arm frozen-backbone comparison from the separate common-checkpoint study with a 5M-token freeze. The historical Stage-1 wiring runs are archived and should not silently define the new claim.

### Q1 — Scientific claim

What should the frozen-wiring experiment establish?

A. Whether each added feedback mechanism can learn useful behavior while the pretrained backbone is held fixed.

B. Whether one architecture is categorically better than the others.

C. Whether feedback architectures improve end-to-end language-model quality.

**Recommended:** A. Frozen wiring isolates conditional controller learnability; it cannot by itself establish full-model superiority or downstream gains.

### Q2 — Exact campaign

Which experiment are we grilling?

A. The current four-arm `frozen_backbone_comparison`: Recirculation, dense Memory Attention, Strided Memory Attention, and Multiscale Memory Attention, frozen for the complete 100M-token run.

B. The `common_checkpoint_comparison`: vanilla plus feedback arms, with feedback frozen for 5M tokens and then fully unfrozen.

C. The historical six-arm Stage-1 wiring campaign.

**Recommended:** A. Treat B as the follow-up that tests whether wiring transfers into continual backbone training; treat C as provenance, not current evidence.

### Q3 — Comparator

What is the primary comparison?

A. Head-to-head adaptation among the four feedback mechanisms, all starting from one common pretrained checkpoint.

B. Feedback versus vanilla, asking whether adding any feedback pathway is worthwhile.

C. Both, with separate claims and separate controls.

**Recommended:** C. The four-arm study can rank mechanisms; a vanilla or parameter-free control is needed before claiming that feedback itself beats ordinary computation.

### Q4 — Definition of success

What outcome determines whether a wiring arm worked?

A. Lowest final held-out NLL at 100M tokens.

B. Fastest early NLL reduction while the backbone is frozen.

C. A pre-specified learning-curve criterion, with final NLL and efficiency reported as secondary views.

**Recommended:** C. The stated question is “how long do they reduce NLL,” so terminal NLL alone throws away the most relevant evidence.

### Q5 — Interpretation boundary

What are you willing to say if an arm wins?

A. “This mechanism wires effectively into this fixed TinyMistral backbone under this optimization protocol.”

B. “This mechanism is intrinsically superior.”

C. “This mechanism will improve downstream or generation performance.”

**Recommended:** A. A win is conditional on initialization, added-module parameterization, learning rate, pass schedule, data stream, and frozen-backbone duration.

### Q6 — Execution contract

Which details are non-negotiable across arms?

The current protocol specifies `batch_size: 16`, `grad_accum_steps: 2`, 32 sequences per optimizer update, seed `1337`, architecture seed `4242`, constant added-module LR `3e-4`, K=2 with probability 0.9 and K=3 with probability 0.1, final-pass-only loss, and one 100,007,936-token trajectory.

A. Freeze this protocol and change nothing without recording a new study.

B. Hardware-tune batch size, K schedule, or learning rate per architecture.

C. Tune until each arm looks competitive, then compare the best runs.

**Recommended:** A. Any change to these fields changes the experiment’s estimand and must become an explicit qualification or ablation.

### Q7 — Go/no-go and replication

What happens before and after the expensive 100M-token runs?

A. Launch immediately, one seed per arm, and interpret the curves afterward.

B. Require forward-mode, CUDA-memory, and matched-LR qualifications first; then run one seed per arm as a screen and replicate only finalists.

C. Require multiple seeds for every arm before any result is considered.

**Recommended:** B. This respects the repository’s promotion gate while acknowledging that one trajectory cannot support strong variance claims.

### User answers

- **Q1:** A – although the possibility of learning useful behaviour is too vague. I prefer the experiment to illustrate that the attention-mechanisms perform on par or better than recirculation, and that sparse attention provides a counter-example to the notion that we need recurrence for state-tracking.
- **Q2:** A, for now.
- **Q3:** C, in different experiments that are complementary.
- **Q4:** C since i would prefer looking at a learning curve for potential patterns. It might be that one learns quicker and trails off quicker.
- **Q5:** A, but there is evidence for more general notions like a weaker version of B or C.
- **Q6:** Neither. We tune LR mostly but we keep effective batch size, K schedule etc. constant.
- **Q7:** A, I prefer to not have too much promotion logic.

## Round 2 — Operationalize the claim

The answers narrow the project to an adaptation-curve comparison, with attention mechanisms expected to match or beat Recirculation and sparse memory access serving as a test against “recurrence is necessary for state-tracking.” The pressure point is that “on par,” “state-tracking,” and LR tuning can each become post-hoc escape hatches.

### Q8 — Define “on par”

How will you decide that an attention mechanism performs on par with Recirculation?

A. It ranks similarly by final NLL.

B. It stays within a pre-declared practical NLL margin over the learning curve and at the endpoint.

C. We inspect plots and make a qualitative judgment.

**Recommended:** B. Pick the margin before seeing the 100M curves. A single final-point ranking misses the early-fast/late-slow pattern you explicitly care about.

### Q9 — State tracking

What evidence would justify the phrase “sparse attention is a counter-example to recurrence being necessary for state-tracking”?

A. Comparable held-out NLL on the ordinary Dolmino language-model stream.

B. Add a small targeted long-range copy/associative-recall diagnostic.

C. Avoid “state-tracking” and say only that recurrence is not necessary for this LM adaptation result.

**Recommended:** B if you want to retain the state-tracking claim. Ordinary NLL cannot distinguish state tracking from generic extra depth, easier optimization, or better access to previous-pass features.

### Q10 — Meaning of recurrence

Are you using “recurrence” narrowly to mean a fixed recurrent feedback edge, or broadly to mean any state carried across passes?

The Strided Memory Attention arm is not recurrence in the narrow architectural sense, but it still carries previous-pass states and lets later tokens address retained memory. Calling it “non-recurrent” without qualification would overstate the result.

A. Narrow claim: content-addressable sparse memory can replace fixed recurrent feedback.

B. Broad claim: no recurrent state is needed at all.

C. Make no recurrence-related claim.

**Recommended:** A. It is both interesting and defensible.

### Q11 — Learning-rate tuning

How should LR tuning relate to the architecture comparison?

A. Select one common stable rate for all four mechanisms, then run the 100M comparison.

B. Select the best rate separately for each mechanism.

C. Do both: common-rate results are primary; per-arm rate sensitivity is secondary.

**Recommended:** C. Per-arm tuning answers “best practical performance,” while the common-rate result preserves the cleaner mechanism comparison.

### Q12 — Validation leakage

The LR qualification and the 100M study use the same `paper_1024` validation split. Is that acceptable?

A. Yes; call it a protocol-selected monitoring split and keep the claim narrow.

B. Use a second untouched evaluation artifact for the final comparison.

C. Select LR from training loss only.

**Recommended:** B. Otherwise the long-run NLL curve is partly evaluated on data that influenced LR selection. If generating a second artifact is too costly, choose A explicitly and do not describe the result as untouched held-out evidence.

### Q13 — Evaluation semantics

The arms train with K=2/K=3 but the current frozen study evaluates with K=4. Which view is primary?

A. K=4 only.

B. Training-matched K=2/K=3 validation is primary; K=4 is a pass-depth stability diagnostic.

C. Report only K=1 or generation quality.

**Recommended:** B. Otherwise the main score partly measures extrapolation to an evaluation pass depth that was not optimized.

### Q14 — Minimal execution discipline

“No promotion logic” should not mean “no validity checks.” What is the minimum launch contract?

A. Launch the runs and inspect failures afterward.

B. Require only manifest validation, one forward/backward preflight per arm, finite loss/nonzero added gradients, zero backbone drift, and auto-resume.

C. Keep the full formal qualification and promotion machinery.

**Recommended:** B. This is a small operational checklist, not a bureaucratic promotion process; without it, a 100M-token run can produce uninterpretable evidence.

### User answers

- **Q8:** C. Nothing is set in stone yet. The experiments are not ready for publication and might/probably will be tinkered with.
- **Q9:** A to begin with. We should test on downstream tasks also, though. 100M tokens is usually too few tokens to gain larger capability changes, so I do not expect much.
- **Q10:** A, and with “recurrence” i pretty much mean time-dependent latent state transfer a la h_{t-1} updates some h_t in an earlier layer. The key is that the connection relies only on the “last timestep latent”, so maybe this is not strictly speaking recurrent, when the latents are not in the same layer.
- **Q11:** I would prefer B with a small initial sweep like we have done for now.
- **Q12:** A, fine i do not care about this minor point.
- **Q13:** B
- **Q14:** B, but with some variation perhaps. I think one should run some model-dependent efficiency checks about batch-size and gradinet-accumulation on a new piece of hardware, so that they can be optimized for running fast while still being the same experiment. However, these should not be obligatory and is something i would like to have the option to tell an agent to run and if not then defaults are used.

## Round 3 — Remove escape hatches

The project is explicitly exploratory, but “on par,” the state-tracking interpretation, and LR selection still need operational definitions.

### Q15 — Protocol versioning

If the study remains intentionally open to tinkering, how do you prevent incompatible runs from being compared as one curve?

A. Edit the configs in place and treat the latest version as authoritative.

B. Keep one exploratory study, but version each protocol change and never merge curves across versions.

C. Freeze the protocol now.

**Recommended:** B. “Not publication-ready” is fine; silently changing the estimand is not.

### Q16 — What the sparse result isolates

Recirculation transfers essentially one fixed, shifted latent state, while Strided Memory Attention retains and addresses up to 32 strided memory records. Multiscale has both dense and sparse records.

Are you comparing:

A. Useful mechanisms as implemented, allowing different state bandwidths and capacities.

B. The access primitive itself, requiring matched retained-state capacity and information bandwidth.

C. Both: screen the useful mechanisms now, then run a capacity-matched mechanistic follow-up.

**Recommended:** C. The current screen can show that sparse addressable memory works well; it cannot by itself show that attention beats recurrence rather than simply providing more accessible state.

### Q17 — Precise recurrence claim

Is the intended claim:

A. Fixed last-timestep latent transfer is not necessary when content-addressable sparse previous-pass memory is available.

B. Time-dependent latent state transfer is not necessary for state tracking.

C. Recurrence is generally inferior to attention.

**Recommended:** A. It preserves the conceptual contrast without pretending that Strided Memory Attention is stateless.

### Q18 — Per-arm LR selection

For each mechanism’s small LR sweep, what rule selects the rate before the 100M run?

A. Lowest final 5M-token validation NLL.

B. A pre-declared combination of final NLL, stability, gradient behavior, and parameter drift.

C. Whichever rate produces the most attractive eventual 100M curve.

**Recommended:** B. Per-arm tuning is defensible, but the rule must be fixed before the long runs and cannot use their outcomes.

### Q19 — Primary validation views

Should the main curve contain:

A. A single mixture score averaging K=2 and K=3 according to the training probabilities.

B. Separate K=2 and K=3 final-pass NLL curves, with K=4 reported as a diagnostic.

C. K=4 as the main curve plus K=2/K=3 in an appendix.

**Recommended:** B. Separate curves reveal whether a mechanism learns one pass depth well and fails at the other.

### Q20 — Downstream evidence

You want downstream tests eventually, but not necessarily as part of this 100M-token screen. What role should they have?

A. Separate follow-up study after the frozen wiring screen, including generation and selected LM-eval tasks.

B. A mandatory gate before interpreting any frozen-wiring result.

C. Optional anecdotal examples only.

**Recommended:** A. Keep the frozen screen focused, but do not let its NLL result silently become a downstream capability claim.

### Q21 — Hardware mode

What should happen when a new GPU supports a different efficient microbatch?

A. Keep the declared 16×2 configuration by default, with optional hardware tuning as a separate engineering mode.

B. Permit different `(batch_size, grad_accum_steps)` pairs when their product preserves the nominal optimizer batch of 32, while keeping data dose, K distribution, LR, precision, clipping, and loss scaling fixed. Record the selected pair.

C. Permit any faster configuration, including changes to the effective optimizer batch.

**Recommended:** A plus B as an explicit opt-in mode. The default remains directly reproducible; the optional mode preserves the nominal optimizer batch and records that accumulation/grouping changed.

### User answers

- **Q15:** B, but realistically i might do A and then arrive at a final version.
- **Q16:** C, but the capacity seems roughly matched, no? As long as the compute budget is close there might just be too large a difference to be worth a follow-up. Im thinking of just adjusting for compute on the same run and scaling appropriately.
- **Q17:** I want to provide evidence for C, not conclusively establish it. A is too weak as the “not necessary” point is kind of trivial to most intelligent people. B is the same.
- **Q18:** A, it is simpler. But maybe I should do B.
- **Q19:** By default I think we should choose K=4 for the eval curve, but this can be a parameter for plotting if we want. K=4 is a diagnostic, but i think it is also representative for the kind of generalization one should expect at longer inference sequences.
- **Q20:** A

## Round 4 — Resolve the remaining conceptual traps

The memory capacity is not obviously matched: Dense and Strided Memory Attention each expose 32 retained records; Multiscale exposes 32 dense plus 32 sparse records; Recirculation exposes one shifted source latent at each destination position. Similar FLOPs do not imply similar state bandwidth or addressability.

### Q21 — Hardware mode

What should happen when a new GPU supports a different efficient microbatch?

A. Keep the declared 16×2 configuration by default, with optional hardware tuning as a separate engineering mode.

B. Permit any `(batch_size, grad_accum_steps)` pair whose product remains 32, and record the selected pair.

C. Allow any hardware-optimized batch and accumulation settings.

**Recommended:** A plus B as an explicit opt-in mode.

### Q22 — Temporary tinkering

If you edit configs in place before reaching the final protocol, what makes the final version identifiable?

A. The final config itself is sufficient.

B. Record a final protocol commit/tag and state which earlier runs are exploratory and incomparable.

C. Rerun every historical configuration after the final edit.

**Recommended:** B.

### Q23 — Capacity-matched follow-up

If the current screen shows Strided Memory Attention matching Recirculation, what result would force you to stop attributing the difference to the access primitive?

A. Strided Memory Attention uses more directly addressable state than Recirculation.

B. A capacity-matched variant removes the advantage.

C. The two arms have similar estimated FLOPs.

**Recommended:** B.

### Q24 — Evidence for the broad C-level story

What minimum combination would count as meaningful evidence?

A. One frozen-backbone Dolmino curve.

B. Frozen-backbone curves plus downstream/generation results and at least one state-tracking-oriented evaluation.

C. A lower final NLL from any attention arm.

**Recommended:** B.

### Q25 — Simple LR selection with a guardrail

Is “lowest 5M-token validation NLL” enough to select each arm’s LR?

A. Yes, exactly as written.

B. Lowest final NLL among trajectories that pass a minimal finite-loss/no-catastrophic-instability check.

C. Use a weighted score involving NLL, drift, throughput, and stability.

**Recommended:** B.

### Q26 — What K=4 actually tests

K=4 is a larger pass depth, not a longer token sequence. What is its role?

A. Headline metric because the intended inference setting uses four refinement passes.

B. Diagnostic for pass-depth extrapolation; evaluate longer token contexts or generation separately.

C. Proxy for longer-context generalization.

**Recommended:** B, unless four-pass inference is explicitly intended.

### Q27 — Compute normalization

When you say “adjusting for compute,” which comparison is primary?

A. Equal linguistic-token dose; report time and estimated FLOPs alongside it.

B. Equal accelerator time.

C. Equal estimated dominant FLOPs.

D. Produce all three views and avoid collapsing them into one adjusted score.

**Recommended:** D.

### Q28 — Follow-up scope

For the later downstream study, which evidence should be prioritized?

A. Standard LM evaluation and generation only.

B. Standard LM evaluation, generation, and one targeted long-range state-tracking task.

C. Only the targeted synthetic task.

**Recommended:** B.

### User answers

- **Q21:** We can set a default, but i would set it lower than 16x2. I would prefer something like 8x4 default and lower if batch-size does not allow for it, and then the other option should be `(batch_size, grad_accum_steps)` pair whose product remains 32 optimization.
- **Q22:** B is good, but i might do A when i feel like the codebase is at the right point.
- **Q23:** I do not understand this point. The strided attention has the same default capacity as the dense attention. So if dense performs better this is because of the access-pattern, not the capacity as such. Im not interested in strided memory attention as such since i think it is too limited, but it would show that performance-increses need not come from aforementioned “recurrence”.
- **Q24:** B.
- **Q25:** Fine, B.
- **Q26:** B, and I do have an eval-suite for generation-aligned testing, no? (check this please).
- **Q27:** C
- **Q28:** A for now but ideally B later.

## Round 5 — Sharpen the comparison and execution modes

The repository does have generation-aligned evaluation:

- `evaluation/suites/generation_math.yaml` covers GSM8K and Hendrycks Math.
- `evaluation/suites/generation_code.yaml` covers HumanEval and MBPP.
- `evaluate_lm_harness.py` supports prompt `--prefill-passes K` plus live `--decode-mode feedback`; generated tokens advance the feedback state once each.

There is not yet a dedicated state-tracking suite.

### Q29 — Role of each attention arm

Given the clarification, should the arms have distinct interpretive roles?

A. Dense Memory Attention is the dense-memory reference, Strided Memory Attention is the sparse non-recurrent existence proof, and Multiscale is the practical hybrid-memory candidate.

B. Treat all three as interchangeable members of one “attention” class.

C. Drop Strided Memory Attention because it is too limited.

**Recommended:** A.

### Q30 — Falsifier for the broad claim

What result would make you weaken the claim that recurrence is generally unnecessary or inferior?

A. Recirculation catches up on the 100M frozen NLL curve.

B. Recirculation wins on downstream generation or state-tracking tasks.

C. Any attention arm wins on any one metric, so the claim is supported.

**Recommended:** B.

### Q31 — Equal-compute analysis

Should the runs be:

A. Trained for the full 100M tokens, then compared at equal tokens, estimated FLOPs, and accelerator time by interpolation.

B. Stopped at different token counts so every arm receives exactly the same estimated FLOPs.

C. Reported only at equal estimated FLOPs.

**Recommended:** A.

### Q32 — Hardware-tuned pair selection

What happens when an arm cannot fit batch 8?

A. Try 8×4, then deterministically fall back through 4×8, 2×16, and 1×32, preserving the 32-sequence optimizer batch.

B. Choose any pair that happens to be fastest.

C. Use a single common fallback pair for all arms on that hardware.

**Recommended:** A, but record the selected pair per arm and hardware.

### Q33 — Protocol freeze trigger

What does “the codebase is at the right point” mean before calling a run the final exploratory version?

A. The implementation feels mature enough.

B. The relevant correctness tests, forward-mode checks, and config semantics are stable; then record a commit/tag and freeze the protocol.

C. The first complete 100M run defines the final version retroactively.

**Recommended:** B.

### Q34 — Generation follow-up

For the later downstream study, should the existing suites be run with:

A. `prefill_passes=4`, `decode_mode=feedback`, plus standard decoding as an ablation.

B. Only ordinary standard decoding.

C. A separate K-pass rerun for every generated token.

**Recommended:** A.

### User answers

- **Q29:** A, kind of, as i think the dense swa pattern is promising on its own. But I would like all models tested and treated as mostly equal citizens for now. I do not really see the point of this question. They
- **Q30:** C I expect the downstream performance and NLL to track roughly. But i do think we need to consider what performance-margins are significant. If the results seem too noisy on generation-evals or if the answers seem largely unaffected as one would perhaps expect after such short training runs, then I will not update much and we will need better experiments.
- **Q31:** A.
- **Q32:** A.
- **Q33:** Mixture of all of them. I will need more experimentation before i can give a good answer.
- **Q34:** A, since option C defeats the reason to have this architecture in the first place. A is the answer and it should always be this. How are we doing evals now btw?

## Round 6 — Evaluation and execution interpretation

The current evaluation setup was checked:

- During frozen-backbone training, validation runs every 3,276,800 linguistic tokens on 64 blocks with `eval_passes: 4`.
- The recorded headline `nll` is the final-pass K=4 NLL; `nll_by_pass` retains K=1 through K=4.
- This is teacher-forced full-sequence NLL, not generation.
- The LR qualification evaluates the same way, but every 1,048,576 tokens.
- `evaluate_pass_depth.py` can separately report K=1 through K=8.
- Later, `evaluate_lm_harness.py` can run the generation suites with `--prefill-passes 4 --decode-mode feedback`.
- There were no retained frozen-wiring result artifacts at that point.

### Q35 — K=4 headline

Given that K=4 is the intended refinement depth, should training-time validation remain:

A. K=4 as the headline NLL, while always retaining K=1–K=4 per-pass values.

B. K=2/K=3 as the headline because they match training, with K=4 secondary.

C. K=4 only, without per-pass reporting.

**Recommended:** A.

### Q36 — Longer-context versus longer-pass testing

Since K=4 tests refinement depth rather than token context length, should the later downstream phase include:

A. Generation-suite evaluation only.

B. Explicit context-length scaling only.

C. Both generation with live feedback and context-length scaling.

**Recommended:** C eventually.

### Q37 — Capacity matching now

Given that Dense and Strided Memory Attention share the same 32-record capacity, while the immediate point is only that a sparse attention route can work without a fixed recurrent edge, should capacity matching be removed from the mandatory experiment tree?

A. Yes; keep it as an optional follow-up only if the Strided-versus-Recirculation result is surprising.

B. No; require it before interpreting any initial screen.

C. Match capacity across all four arms immediately.

**Recommended:** A.

### Q38 — Broad-claim update rule

If NLL improves clearly but generation results are flat or noisy after 100M tokens, what should be concluded?

A. The NLL result still supports the broad claim.

B. The result is evidence for improved teacher-forced adaptation, but not yet for downstream capability.

C. The broad claim is falsified.

**Recommended:** B.

### Q39 — Hardware configuration ownership

With 8×4 as the preferred default, where should the selected pair live?

A. In a recorded hardware-profile/config override, leaving the scientific arm configs unchanged.

B. By editing each scientific arm config directly.

C. Only in the launcher’s transient command line.

**Recommended:** A.

### Q40 — “Final” exploratory version

If A is sometimes used—editing configs in place—what event should cause the current files to stop being provisional?

A. When the implementation and tests are stable enough for a meaningful run, record the commit and call that protocol version final.

B. Only after the results look persuasive.

C. Never; all versions remain one continuous experiment.

**Recommended:** A.

### User answers

- **Q35:** A, but I am wondering if we should have the option to do teacher-forced feedback mode for the normal eval also.
- **Q36:** C, I think, but it might not be a priority now.
- **Q37:** I think A, but I do not know what you mean about capacity matching.
- **Q38:** B
- **Q39:** B
- **Q40:** A

## Round 7 — Separate evaluation semantics and claim strength

The repository’s `evaluate_recurrent_inference.py` already provides a teacher-forced continuation diagnostic. It prefills a prompt with exact K-pass inference, feeds the observed continuation tokens one by one, compares exact K-stream, collapsed feedback, and vanilla K=1 continuation, and reports NLL by offset and cumulative horizon plus hidden-state drift.

### Q41 — Teacher-forced feedback evaluation

Should feedback-mode evaluation be added to the normal evaluation package?

A. Keep K=4 full-sequence NLL as the main curve, and add teacher-forced live-feedback continuation as a separate diagnostic.

B. Replace full-sequence NLL with feedback-mode NLL.

C. Combine both into one score.

**Recommended:** A.

### Q42 — Feedback diagnostic scale

The current continuation diagnostic uses 256 prompt and 256 continuation tokens. Should later evaluation:

A. Keep those defaults for inexpensive diagnostics.

B. Add longer horizons and prompt lengths as a separate scaling study.

C. Replace the defaults with the longest feasible context immediately.

**Recommended:** B.

### Q43 — Capacity-matching decision

Dense versus Strided Memory Attention is a reasonably clean access-pattern comparison because both expose 32 records. Recirculation versus Strided Memory Attention is not capacity matched: Recirculation receives one shifted source latent at the destination, while Strided Memory Attention can attend over up to 32 previous-pass records.

Do we interpret the initial screen as:

A. A performance study of useful mechanisms, postponing causal attribution between access bandwidth and recurrence.

B. A mechanistic test that requires a matched Recirculation/Memory-Attention state budget first.

C. A direct proof that attention beats recurrence whenever Strided wins.

**Recommended:** A.

### Q44 — Significant margins

Since fixed thresholds are not desired yet, how should noisy differences be handled?

A. Rank every observed difference, however small.

B. Use practical effect sizes and uncertainty estimates, while treating one-seed results as exploratory.

C. Ignore uncertainty until downstream evaluations are available.

**Recommended:** B.

### Q45 — Batch/accumulation logging

The trainer already records `microbatch_tokens`, `accumulation_steps`, nominal optimizer-batch tokens, throughput, and the resolved config. Is any extra hardware logging needed?

A. No extra subsystem; retain the resolved config and perhaps one human-readable hardware-profile label.

B. Add a dedicated database of all hardware experiments.

C. Do not record the selected pair.

**Recommended:** A.

### Q46 — Timing the downstream suite

How should we estimate whether downstream evaluation is affordable?

A. Run the complete math and code suites immediately on every checkpoint.

B. Time a small `--limit` pilot with a representative checkpoint and fixed `--max-gen-tokens`, then decide whether full evaluation is affordable.

C. Estimate runtime from the number of tasks without running it.

**Recommended:** B.

### User answers

- **Q41:** A.
- **Q42:** I mean should we not match the train and test distribution and use full length always?
- **Q43:** A. For now we discuss mechanisms. One of the benefits of attention is ofcourse larger capacity. but we cannot isolate this variable as it would require testing a thousand different mechanisms.
- **Q44:** A mixture of B and C. If there is little downstream difference, then the pratical effect-sizes must be relatively small.
- **Q45:** A, maybe... But does this need to be logged? With gradient-accumulation i do not think this is important, so i do not want to over-engineer.
- **Q46:** I want to see how long the suite takes first.

## Round 8 — Align evaluation with the data

The current training-time NLL evaluation already uses full 1,024-token packed blocks from `paper_1024`. The 256+256 setting belongs only to the separate feedback-continuation diagnostic.

For feedback evaluation, “full length” means the prompt plus observed continuation should consume the full block. It does not mean rerunning the full K-pass computation for every token.

### Q47 — Full-length feedback diagnostic

What should the default teacher-forced feedback diagnostic use?

A. A full block, split into prompt and continuation—for example 512 prompt + 512 observed continuation—with cumulative horizon reporting.

B. The current 256+256 short diagnostic.

C. A full prompt plus a continuation from a longer-context artifact.

**Recommended:** A.

### Q48 — Two kinds of NLL

Should reporting keep these separate?

A. Full-sequence K=4 NLL as the main training-adaptation curve, plus full-block teacher-forced feedback NLL as a continuation diagnostic.

B. Replace the full-sequence curve with feedback NLL.

C. Average the two.

**Recommended:** A.

### Q49 — Flat downstream results

If generation differences are small, how will you distinguish “small practical effect” from “underpowered/noisy evaluation”?

A. Treat all small differences as practically irrelevant.

B. Report effect sizes and uncertainty, while explicitly allowing the conclusion that the suite is underpowered.

C. Ignore generation results until a longer run exists.

**Recommended:** B.

### Q50 — Batch/accumulation logging

Is any extra hardware logging needed?

A. No extra subsystem; retain the resolved config and perhaps one human-readable hardware-profile label.

B. Add a dedicated database of all hardware experiments.

C. Do not record the selected pair.

**Recommended:** A.

### Q51 — Timing the downstream suite

How should runtime be estimated?

A. Run the complete math and code suites immediately on every checkpoint.

B. Time a small pilot, then decide ad hoc whether to use better hardware, validate less often, or change the evaluation plan.

C. Estimate runtime without running it.

**Recommended:** B.

### Q52 — Evaluation priority after the protocol switch

Once the 2,048 study is running, should we:

A. Keep full-sequence K=4 NLL as the main curve, add BOS-only feedback evaluation only if inexpensive, and defer downstream generation until useful checkpoints exist.

B. Make feedback NLL the new main curve.

C. Run all evaluation modes on every checkpoint regardless of cost.

**Recommended:** A.

### User answers

- **Q47:** I think we do no context for this when it is the case that there is not a question and an answer? Can you give me context here? This eval is just on a held-out part of the training data, not a downstream task, no?
- **Q48:** A
- **Q49:** A until we get something more significant, then we can do B.
- **Q50:** A
- **Q51:** We do not need a predetermined limit for this... we will figure it out and decide if we need better hardware, validate more rarely or what we will do. Well just run something and check in an ad hoc manner. Please stick to important questions and not little details like this. That is a waste of my time.
- **Q52:** If the feedback validation is not too costly then we add it.

## Round 9 — Resolve the sequence and evaluation contract

At the time of this exchange, the current frozen-wiring configs did not train on 2,048-token chunks. They used `data/dolmino/paper_1024`. The repository then also contained 2,048-token artifacts: `gpu_2048`, `gpu_2048_staged`, `gpu_2048_long_2p5b`, and `wiring_2048`.

The feedback diagnostic was clarified as held-out ordinary text rather than downstream QA. A prefix/continuation split can be arbitrary in a language-model stream; the question is whether online teacher-forced feedback reproduces the intended generation semantics.

### Q53 — Training block length

Which sequence length is intended for the frozen-wiring study?

A. Keep the current active 1,024-token `paper_1024` contract.

B. Switch the frozen study to 2,048-token chunks to match the historical wiring setup.

C. Run both as separate exploratory protocols.

**Recommended:** A for the current study unless long-context behavior is itself central. If choosing 2,048, make it a separately versioned study.

### Q54 — Feedback validation role

Given that this is held-out ordinary text rather than a downstream task, should we add it only as:

A. An optional semantic diagnostic on selected checkpoints, if its cost is acceptable.

B. A required second headline curve for every training snapshot.

C. Something to defer entirely until downstream generation.

**Recommended:** A.

### Q55 — Full-block feedback split

If using the current artifact, should the diagnostic consume the full block as prefix plus continuation?

A. Yes; use a fixed split such as 512 prefix + 512 observed continuation and report cumulative horizons.

B. Keep the existing 256+256 diagnostic because it is cheaper.

C. Use an arbitrary split per checkpoint.

**Recommended:** A.

### Q56 — Claim boundary

Until downstream and context-scaling work exists, should the frozen screen be described as evidence that:

A. Attention-based feedback mechanisms can match or exceed Recirculation under this fixed LM training protocol.

B. Attention generally beats recurrence.

C. Recurrence is unnecessary for state tracking.

**Recommended:** A, while treating the observed pattern as motivation for the broader hypothesis.

### User answers

- **Q53:** B. We should test on longer horizons. Perhaps we should sweep lr 1e-4 to account for the larger number of tokens in each example. And we should remove the stale 1024 version. We do have a block-length of 2048 for the longer runs also, no?
- **Q54:** A
- **Q55:** I think we do pure teacher-forced decoding and no prefill. When we do have a question to prefill this is another matter. In this case we are testing a generation-mode with teacher-forcing. It should work with and without context-prefill.
- **Q56:** A and C although C should be obvious.

## Round 10 — Final decisions before implementation

At the time of this exchange, the repository facts were checked:

- `gpu_2048` was a 100M-token, 2,048-token serious-run artifact.
- `gpu_2048_staged` was a 100M-token 2,048-token artifact with a 5M-token offset.
- `gpu_2048_long_2p5b` was a 2.5B-token 2,048-token continuation artifact.
- `wiring_2048` was a 5M-token 2,048-token artifact.
- The development frozen-wiring studies were then inconsistent because they still pointed to `paper_1024`.

### Q57 — 2,048 data artifact

Which artifact should the new frozen-wiring study consume?

A. `gpu_2048`, with 100M training tokens and its 2M-token validation split.

B. `wiring_2048`, repeatedly cycling the 5M wiring slice.

C. `gpu_2048_staged`, preserving the 5M-token offset used by the historical staged pipeline.

**Recommended:** A.

### Q58 — LR at longer sequence length

If 2,048-token blocks halve optimizer updates per linguistic token under the same 32-sequence optimizer batch, how should LR qualification change?

A. Re-run a small 2,048-specific LR sweep around the existing values, then choose per-arm using the same simple final-NLL-plus-stability rule.

B. Automatically halve or double the LR based on sequence length.

C. Keep the 1,024-token LR result unchanged without checking.

**Recommended:** A.

### Q59 — Pure feedback validation

For held-out ordinary text with no question/answer prompt, should the online diagnostic be:

A. BOS-only initialization followed by teacher-forced feedback through the entire 2,048-token block.

B. A fixed prompt prefix plus teacher-forced continuation.

C. Full-sequence parallel K-pass evaluation only.

**Recommended:** A for the pure online-semantic diagnostic. Use B separately for prompt-conditioned generation behavior.

### Q60 — Evaluation priority after the protocol switch

Once the 2,048 study is running, should we:

A. Keep full-sequence K=4 NLL as the main curve, add BOS-only feedback evaluation only if inexpensive, and defer downstream generation until useful checkpoints exist.

B. Make feedback NLL the new main curve.

C. Run all evaluation modes on every checkpoint regardless of cost.

**Recommended:** A.

### User answers

- **Q57:** A but validating on 2M tokens seems expensive. We could just do it rarely. But we need to stop grilling this point and create experiments instead so we can see what is feasible.
- **Q58:** A, rerun the lr sweep with larger choices included. Perhaps we consider 1e-3 also?
- **Q59:** A
- **Q60:** A

## Implementation outcome

### Assistant

We have enough shared understanding to act. The frozen-wiring comparison and its LR qualification will be converted to the 2,048-token `gpu_2048` artifact, use 8×4 as the default 32-sequence optimizer batch, add a 1e-3 LR candidate, and keep the 2M-token validation available for infrequent full evaluations rather than every training check. The separate common-checkpoint and forward-policy studies remain untouched because they are different protocols.

### Changes made

- Frozen comparison configs now use `data/dolmino/gpu_2048`.
- Frozen comparison configs now use `batch_size: 8` and `grad_accum_steps: 4`.
- The LR qualification configs use the same 2,048-token and 8×4 setup.
- Four 1e-3 LR arms were added, bringing the qualification study to 16 arms.
- Frozen-wiring documentation described the 2,048-token protocol, provisional per-mechanism LR selection, optional full validation, and the new batching policy.
- Stale tests were updated to distinguish the 2,048-token frozen-wiring studies from the separate 1,024-token studies.
- `paper_1024` was retained because it was then used by the separate common-checkpoint and forward-policy studies.

### Verification

- Both updated study manifests validate.
- The focused study/config/runner tests passed.
- The full test suite passed: **360 passed, 10 skipped** because MPS was unavailable.
- `git diff --check` passed.
- `make estimate-flops-frozen-backbone` succeeded.

### Final implementation summary

The updated protocol is documented in `benchmarks/development/frozen_backbone_comparison/README.md`.

The LR qualification can be run with:

```bash
uv run python scripts/run_study.py \
  --study-dir benchmarks/development/frozen_backbone_lr_qualification \
  --skip-wire
```

### User

Can you give me the contents of our exchange as a markdown file please?
