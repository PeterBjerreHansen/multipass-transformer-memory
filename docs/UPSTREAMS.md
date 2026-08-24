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

Memory Attention and Sparse SWA add two tested substrate capabilities. Self-attention K/V
entries can carry a validity mask, and selected layers can use a bounded
dense-recent/fixed-periodic-old mask. Write-only `<MEM>` positions retain their
physical, RoPE, and cache coordinates while remaining unavailable as K/V. An
empty attention row returns exact zero in every backend. Ordinary layers retain
the original local path, while Sparse SWA reuses the pretrained projections.

## FBT architecture reference

The `fbt` variant is based on the asymmetric latent-feedback construction in
Xi Wang et al., *Full-bandwidth Transformer*, arXiv:2608.08888. This repository
uses a TinyMistral retrofit and does not claim to reproduce the paper's full
pretraining recipe.

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
