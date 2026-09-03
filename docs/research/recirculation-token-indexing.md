# Recirculation: token indexing

> Cleanup update (2026-09-03): paper readout/replay and BPTT/TBPTT execution
> and the old 1024-token proposals have now been deleted.
> Ordinary preceding-token feedback inference remains supported. BOS-only
> prefill is an input choice for that mode, not a deleted special mode. Earlier
> statements below about deferral, unchanged manifests and local paper-method
> checks describe the research state before cleanup. See
> [CLEANUP_STATUS.md](../CLEANUP_STATUS.md) for the current decision ledger.

Checked 2026-09-02 against [Mozer et al., *Recirculation*, arXiv:2608.17981v2](https://arxiv.org/html/2608.17981v2).

## Paper

Equation (1) mixes source and destination at the same input position; its update-step index advances. Figure 3 places readout before the additional iteration, which overlaps next-token processing. Adaptive Equation (3) preserves this indexing despite BPTT training (Appendix D.5). No direct comparison with previous-token-to-current-token injection was found in the reported experiments; Section 4.4.3 compares against depth looping. These findings concern the published specification, not uninspected author code. [Methods](https://arxiv.org/html/2608.17981v2#S2), [adaptive training](https://arxiv.org/html/2608.17981v2#A4.SS5).

## This repository

Active parallel recirculation mixes the preceding position's previous-pass source into the current position's destination: see `_run_feedback_state` in [recirculation.py](../../src/tiny_mistral_mptt/variants/recirculation.py) and `shift_previous_hidden` in [multipass.py](../../src/tiny_mistral_mptt/variants/multipass.py). Online feedback likewise reads carried state before replacing it with the newly processed token's source.

User decision on 2026-09-02: retain preceding-token mixing and do not introduce same-token mixing. Drop the proposed separate special feedback inference mode. Keep the paper-replay BPTT/TBPTT experiment and training mode deferred because training is too slow. Study manifests have not been changed.

## Training and efficiency

The paper requires serial prefill (§2) but claims low generation overhead from overlapping two stacks (§5). It proposes blockwise recurrence as unexplored future work (§5.2). [Execution and limitations](https://arxiv.org/html/2608.17981v2#S5).

Adaptive training uses BPTT with the backbone frozen, except in the full-fine-tuning comparison. [Appendix D.5](https://arxiv.org/html/2608.17981v2#A4.SS5).

No whole-block multi-pass training result or speed comparison against preceding-token injection was found in the paper. Its generation-efficiency claim therefore does not establish superiority over this repository's shifted feedback.

## Local gradient checks

On a tiny randomly initialized, frozen-backbone model, the current serial replay path produced nonzero controller gradients. Detaching every token's returned KV cache preserved logits exactly but left the language-model loss with no gradient path. The controller affects later predictions through that cache; freezing backbone parameters does not remove the need to differentiate through its operations. This agrees with the existing earlier-cache gradient test in [test_recirculation.py](../../tests/test_recirculation.py).

A diagnostic-only subclass removed the source shift from the parallel hook. Its K=2 loss trained the controller, but its logits differed from serial replay, including at the first position. Thus same-position adaptive mixing can be trained through parallel passes, but changing the source alignment alone does not reproduce the serial readout/replay policy. Neither check measures training throughput or model quality. No model implementation was changed.

## Relation to looping

The paper's fixed-depth looping comparator completes repeated layers before readout. Recirculation reads out first, then makes replayed information available at shallower depth to subsequent tokens. The distinction is the cross-token state-update path, not weight reuse or same-token mixing alone. This comparison does not cover every architecture called a looped transformer. [Section 2, Figures 3–4](https://arxiv.org/html/2608.17981v2#S2).

In this repository's fixed-K parallel graph, each explicit shifted-feedback edge advances the pass index. A path therefore has at most K-1 such edges; attention can still reach a longer context. This is bounded-depth multi-pass computation with shifted connections, not the continuing temporal loop of single-stream feedback. Without that decoding mode, the benchmark compares feedback mechanisms under the common fixed-K policy; it does not directly test the paper's temporal-recurrence argument. See `_run_passes` in [multipass.py](../../src/tiny_mistral_mptt/variants/multipass.py).
