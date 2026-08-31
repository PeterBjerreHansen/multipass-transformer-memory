# Stage 5 results

The repository README contains the compact eight-arm result table. All eight
arms are complete. Generated run artifacts belong directly under ignored
`<arm>/` directories; each completed arm includes a checksum manifest produced
by the cloud transfer controller.

The retained `evaluation_full_mps.json` files were produced before checkpoint,
config, suite, seed, and sample-level evidence became mandatory. Several are
numerically identical and cannot be audited from the files alone. The legacy
adapter also inherited fixed 512-token padding/truncation from `tokenizer.json`,
which could reduce lm-eval candidate continuations to empty encodings and return
zero log-likelihoods. Treat these files as invalid capability evidence. Rerun
the milestone snapshots with `scripts/evaluate_lm_harness.py`; do not overwrite
the historical files.
