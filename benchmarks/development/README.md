# Development benchmarks

Use this directory for structured studies that inform the protocol without
being the final claim-establishing campaign. A development study should exist
only when it answers a protocol question worth retaining.

Stages 0 through 4 contain the development protocol. The locked eight-arm
Stage-5 core study contains the two attention controls alongside the original
six arms:

- `stage_0_implementation_gates/`: code, test, and wiring preflight gates.
- `stage_1_wiring/`: local Phase-A adaptation of added feedback pathways.
- `stage_2_local_smoke/`: 1M-token local Phase-B integration checks.
- `stage_3_cloud_pilot/`: resumable 5M/10M seed-1337 cloud pilots.
- `stage_4_confirmation/`: two additional seeds for promoted arms.
- `recirculation_forward_qualification/`: paper-forward BPTT versus the current
  whole-block recirculation objective at the paper's 100-step endpoint.
- `frozen_backbone_curves/`: planned 20M-token controller-only curves and
  recurrence-policy comparison.
- `common_checkpoint_retrofit/`: planned single-trajectory 100M comparison with
  a 5M-token integrated backbone freeze for feedback arms.

The last three are planned core candidates. They remain in `development/`
until their hardware-facing microbatch, BPTT/TBPTT window, and learning-rate
choices are qualified and the manifests can be locked.

Pass-depth stability, memory interventions, and exact-vs-recurrent drift are
reusable checkpoint diagnostics rather than separate development studies. Run
them from `scripts/` whenever a validation round needs those measurements.
