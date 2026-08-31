# Core benchmarks

This directory is reserved for reviewed, predeclared studies used to support
the paper's central claims. It is intentionally empty while the three current
study candidates remain under `../development/` for forward-policy, hardware,
batching, and learning-rate qualification.

A promoted core study contains `README.md`, `STUDY.yaml`, runnable arm configs,
and `results/`. Set `status: locked` only after the scientific question, arms,
comparison axes, data artifact, initialization, optimizer batch, learning
rates, BPTT/TBPTT policy, and evaluation semantics have been reviewed. Run
`scripts/verify_study.py` before execution.

Completed studies from the superseded staged protocol are preserved under
`../historical/staged_pipeline/`; their previous location here does not make
them part of the current contract.
