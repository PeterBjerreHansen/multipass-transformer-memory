# Frozen-backbone comparison

This is a planned core candidate, not a locked campaign. All trained arms keep
the pretrained TinyMistral backbone frozen for the complete 20,021,248-token
trajectory. The initialized common checkpoint is the token-zero reference; it
is evaluated but is not a fake training arm.

The snapshots align to an effective optimizer batch of 32 sequences. The first
snapshot, 3,276,800 input tokens, is exactly 100 optimizer steps and matches the
paper's training endpoint. Later snapshots test whether each added mechanism
continues improving through 20M tokens.

The BPTT arm uses recurrent teacher-forced validation. The parallel arms retain
whole-block pass-depth validation as a diagnostic. Before comparing curves,
evaluate every shared snapshot under both applicable semantic views; K is not a
parameter of the token-diagonal recurrence.

Promote this directory to `benchmarks/core/` and set `status: locked` only after
the forward-mode, CUDA-memory, truncation-window, and learning-rate
qualifications are complete.
