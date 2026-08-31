# Forward-policy qualification

This planned study qualifies the two Recirculation training policies before
either can enter a core comparison. `recirculation_bptt` is the paper-style
policy: each
token is read out normally, replayed from the source/destination mixture, and
the replayed upper-layer KV state is used by the next token. It is not K=2.

Both configs use microbatch 1 and accumulation 32 to reproduce
the paper's effective batch of 32 sequences on smaller GPUs. Change those two
fields together if hardware permits a larger microbatch. Preserve their product
unless the optimizer-batch change and learning-rate retuning are explicitly
treated as a new qualification.

The checked-in BPTT arm uses full-sequence gradients and activation
checkpointing. First run `benchmarks/efficiency/suites/forward_modes.yaml` on
the target GPU. If full BPTT is impractical, qualify a finite
`recirculation_bptt_truncate_tokens` value against full BPTT on shorter pilot
runs. TBPTT preserves the forward KV trajectory but changes the gradient path,
so record it as an experimental setting rather than a transparent memory flag.

Run the initialized adaptive model before training as the fixed-mixture
reference: its zero-initialized output projection yields alpha=0.1 and beta=0.9.
Report recurrent teacher-forced NLL for the BPTT policy and retain K=1..8
whole-block NLL only as a separate convergence diagnostic.

The initialized fixed-mixture measurement is intentionally absent from
`STUDY.yaml`: it is an evaluation of the shared token-zero model, not a runnable
training arm.
