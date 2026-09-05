# Results

Retain aligned snapshot learning-curve summaries here. Raw artifacts for each
arm belong under `results/<arm>/` and remain ignored.

The sibling `STUDY.yaml` defines separate one-site and two-site dense groups,
attention-layout extensions, and a two-site stride-length ablation. Plotting and
tabulation must discover those arm IDs and comparison groups instead of pooling
them into one hard-coded arm list. The parameter-free `strided_self_attention`
control is not a Phase-A wiring arm.

Paper replay/BPTT execution and its forward-policy study were deleted.
They are not part of this curve set or FLOP report.

`make estimate-flops-frozen-backbone` writes the ignored, reproducible
`training_flops.json` report here. Join its per-token estimate to the aligned
validation records by `unique_tokens_seen`. The paper-facing summary should
contain NLL versus tokens, optimizer updates, cumulative training-only seconds,
and cumulative estimated FLOPs, plus separately labelled end-to-end wall time
and peak VRAM.

`make report-wiring-budgets` writes `wiring_budgets.json` here. It instantiates
every arm on the meta device to count real added parameters and records stride
write counts, memory spans, K sampling, native initialization, and estimated
FLOPs.
