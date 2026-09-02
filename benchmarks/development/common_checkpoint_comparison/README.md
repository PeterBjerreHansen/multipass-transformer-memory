# Common-checkpoint comparison

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

All arms use 1,024-token ordinary packed blocks, microbatch 8, and accumulation
4 for an effective optimizer batch of 32 sequences. Batch 16 does not fit the
fully trainable dense-memory K=3 path on the target A6000; batch 8 is the common
qualified policy. The token-diagonal arm uses window-128 TBPTT, activation
checkpointing, and reference cached attention. Full BPTT is the paper-faithful
gradient reference but is excluded from the active long run as operationally
infeasible; the finite window remains a separately reported approximation.

Run each arm once. Derive loss-vs-input-token, supervised-target-token,
estimated-FLOP, and accelerator-time views from the same trajectory and retained
snapshots. Do not create separate runs merely to change the x-axis.

The data recipe intentionally remains ordinary packed text. There is no
document-contained view in this iteration.
