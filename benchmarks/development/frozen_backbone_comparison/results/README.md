# Results

Retain aligned snapshot learning-curve summaries here. Raw artifacts for each
arm belong under `results/<arm>/` and remain ignored.

The wiring loss-curve set is defined by the five arms in the sibling
`STUDY.yaml`; plotting and tabulation should discover those arm IDs rather than
maintain a second hard-coded list. In particular, include both
`strided_memory_attention_multipass_20m` and
`multiscale_memory_attention_multipass_20m`. The parameter-free
`strided_attention` control is not a Phase-A wiring arm.

`make estimate-flops-frozen-backbone` writes the ignored, reproducible
`training_flops.json` report here. Join its per-token estimate to the aligned
validation records by `unique_tokens_seen`. The paper-facing summary should
contain NLL versus tokens, optimizer updates, cumulative training-only seconds,
and cumulative estimated FLOPs, plus separately labelled end-to-end wall time
and peak VRAM.
