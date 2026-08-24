# Retired 10M-parent NMP campaign

The local `benchmarks/ad_hoc/recirculation_tape_nmp` campaign was retired on
2026-08-24 after the completed 100M checkpoints became available. Its raw local
tree occupied approximately 46 GB and was not part of the current tracked tree.

## What was completed

The parent was an adaptive-Recirculation + periodic-C32 Memory-Attention hybrid
continued for 10,485,760 NTP tokens. Nine matched 1,048,576-token continuations
used the same parent, data, seed, optimizer restart, fixed data budget, and pass
schedule. Their held-out NTP NLLs were:

| Arm | NLL | Delta versus NTP control |
| --- | ---: | ---: |
| recurrent 5% | 2.198711 | -0.000141 |
| NTP control | 2.198852 | 0 |
| recurrent 10% | 2.198993 | +0.000141 |
| dual 10% | 2.199107 | +0.000255 |
| recurrent 20% | 2.199590 | +0.000738 |
| Memory-Attention 5% | 2.199597 | +0.000745 |
| Memory-Attention 10% | 2.199877 | +0.001025 |
| dual 20% | 2.199877 | +0.001026 |
| Memory-Attention 20% | 2.202922 | +0.004071 |

Training-batch recurrent and Memory-Attention NMP losses fell by approximately
26% and 27–29%, respectively, mostly during the first few hundred thousand
tokens. They were not held-out NMP measurements, so this is evidence of
optimization rather than generalization.

Post-head-warm-up shared-gradient calibration measured NTP norm `1.205069`,
recurrent NMP norm `0.016723`, and Memory-Attention NMP norm `0.070803`.
Coefficients targeting 5% of the NTP shared-gradient norm were `3.603070` and
`0.851007`; the 20% coefficients were `14.412281` and `3.404027`.

The planned 5M campaign did not complete: the NTP control stopped at 4,456,448
tokens and the NMP arms produced no serious-run results. Small ARC/BoolQ screens
tied exactly and GSM8K was at floor for every screened checkpoint; those
capability results were non-discriminating.

## Why it is not the new baseline

The parent had only 10M continuation tokens, validation measured NTP but not
NMP, the objective used the shared final-pass online target, and the serious
matched campaign was incomplete. These results remain useful for choosing a
conservative initial pressure, but they are not evidence that NMP improves the
100M model.
