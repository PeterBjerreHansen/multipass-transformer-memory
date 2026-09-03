# Provenance and pinned external references

## Vanilla backbone

The vendored `src/tiny_mistral/` package is derived from:

- repository: `PeterBjerreHansen/TinyMistralFork`
- validated model-source commit: `e44420d4190b6cfc1dc002c0ac67e364ef2f2de1`
- target checkpoint: `M4-ai/TinyMistral-248M-v3`
- checkpoint revision: `5afbc96ddc964c68282cd970ef49e8d1a5e81c52`
- local checkpoint path: `checkpoints/TinyMistral-248M-v3`
- Transformers architecture oracle: `4.45.2`

Research infrastructure must not change ordinary vanilla behavior silently.
Baseline tests, HF-comparison scripts, and `VANILLA_SOURCE.sha256` are the
guardrails.

The terminology migration changes local K/V variable names and comments in the
attention substrate, without changing its operations. The source manifest records
these edits; numerical-equivalence tests still check behavior independently.

The local cache API additionally accepts `detach_cache`. Its default remains
`true`, preserving ordinary inference and multipass behavior. This generic
substrate capability remains unchanged; the paper replay/BPTT caller has been
deleted. Vanilla numerical-equivalence tests remain authoritative.

Memory Attention and Strided Self-Attention add two tested substrate capabilities. Self-attention K/V
entries can carry a validity mask, and selected layers can use a bounded
dense-recent/fixed-stride-old mask. Write-only `<MEM>` positions retain their
physical, RoPE, and cache coordinates while remaining unavailable as K/V. An
empty attention row returns exact zero in every backend. Ordinary layers retain
the original local path, while Strided Self-Attention reuses the pretrained projections.

## FBT architecture reference

The `fbt` variant is based on the asymmetric latent-feedback construction in
Xi Wang et al., *Full-bandwidth Transformer*, arXiv:2608.08888. This repository
uses a TinyMistral retrofit and does not claim to reproduce the paper's full
pretraining recipe.

## Recirculation architecture reference

The adaptive mixing rule is based on Michael C. Mozer et al., *Recirculation*,
arXiv:2608.17981v2. The active retrofit uses preceding-token, previous-pass
feedback and a late emitted memory. It does not reproduce the paper
readout/replay computation. That execution policy and its BPTT/TBPTT training
implementation have been deleted. The source-indexing research note and
retired protocol records remain provenance, not executable specifications.

## Earlier MPTT research reference

`PeterBjerreHansen/multi-pass-transformer-training` at
`79398be4ac33a7489029e6075bdce930a0ec44b2` is a design reference for
previous-pass top-state feedback, strict recurrence causality, retired
MemoryAdd, and per-layer cross-pass Memory Attention. Current implementations are
written directly against this repository's Mistral/GQA/local-attention
interfaces.

## Training data

- dataset: `allenai/dolmino-mix-1124`
- checked-in recipe revision: `1c2f43706986135c6799d9917e0d06ecef7fb1bb`

No dataset contents are redistributed. Generated manifests record the resolved
revision and artifact hashes.

## Evaluation harness

- project: `EleutherAI/lm-evaluation-harness`
- package pin: `lm-eval==0.4.12`
