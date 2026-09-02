# Forward-policy qualification

This planned study qualifies the two Recirculation training policies before
either can enter a core comparison. `recirculation_bptt` uses the paper-style
token-diagonal forward: each
token is read out normally, replayed from the source/destination mixture, and
the replayed upper-layer KV state is used by the next token. It is not K=2.

Both configs use microbatch 16 and accumulation 2, retaining an effective
optimizer batch of 32 sequences. This is the largest common policy qualified
for all frozen-backbone mechanisms on the target A6000. It amortizes much of
the token-serial TBPTT launch overhead while keeping the physical batch
controlled across the comparison.

The checked-in token-diagonal arm uses window-128 TBPTT and activation
checkpointing. Full BPTT remains the paper-faithful gradient reference, but it
is not an active long-run arm: the audited implementation requires a serial
1,024-token prefill and cannot complete the planned trajectories at a useful
rate. TBPTT preserves the forward KV trajectory but changes the gradient path,
so the window is part of the arm name and reported protocol.

Run the initialized adaptive model before training as the fixed-mixture
reference: its zero-initialized output projection yields alpha=0.1 and beta=0.9.
Report recurrent teacher-forced NLL for the BPTT policy and retain K=1..8
whole-block NLL only as a separate convergence diagnostic.

The initialized fixed-mixture measurement is intentionally absent from
`STUDY.yaml`: it is an evaluation of the shared token-zero model, not a runnable
training arm.
