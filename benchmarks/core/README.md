# Core benchmarks

This directory is reserved for reviewed, predeclared studies used to support
the paper's central claims. It is empty while the frozen comparison and its LR
qualification remain planned under `../development/`.

A promoted core study contains `README.md`, `STUDY.yaml`, runnable arm configs,
and `results/`. Set `status: locked` only after the scientific question, arms,
comparison axes, data artifact, initialization, optimizer batch, learning
rates, parallel-pass objective, and evaluation semantics have been reviewed. Run
`scripts/verify_study.py` before execution. See the [study schema and promotion
rules](../README.md).

Completed studies from the superseded staged protocol are preserved under
`../historical/staged_pipeline/`; their previous location here does not make
them part of the current contract.
