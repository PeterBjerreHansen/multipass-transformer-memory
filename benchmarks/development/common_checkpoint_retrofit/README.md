# Common-checkpoint retrofit

This planned study replaces a separate wiring checkpoint with one continuous
trajectory from the same pretrained TinyMistral backbone. Feedback arms train
only their added parameters through 5,013,504 input tokens, then unfreeze the
backbone without rebuilding the optimizer. The vanilla arm is unfrozen from
token zero.

The optimizer contains the future backbone group from the start. During the
retrofit window it receives no gradients or AdamW steps, while the added group
accumulates its normal optimizer history. At the boundary, the added group's
moments and step count continue unchanged. The LR multiplier stays flat through
the boundary and then decays on the same token-indexed schedule.

All arms use 1,024-token ordinary packed blocks and an effective optimizer batch
of 32 sequences. Microbatch 1 plus accumulation 32 is only the checked-in CUDA
starting point: choose the largest reliable microbatch on the target GPU and
reduce accumulation so their product stays 32. Learning rates and any TBPTT
window must be qualified before this study is promoted to `benchmarks/core/`.
Full BPTT (`recirculation_bptt_truncate_tokens: null`) is the paper-faithful
default; a finite window is a separately reported approximation.

Run each arm once. Derive loss-vs-input-token, supervised-target-token,
estimated-FLOP, and accelerator-time views from the same trajectory and retained
snapshots. Do not create separate runs merely to change the x-axis.

The data recipe intentionally remains ordinary packed text. There is no
document-contained view in this iteration.
