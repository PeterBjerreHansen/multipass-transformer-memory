# Legacy result location

Active frozen-backbone results belong under the tier that produced them:
`small/results/<arm>/`, `medium/results/<arm>/`, or `large/results/<arm>/`.
Reports for the retired 15-arm manifest were removed. Generate reports inside
the selected tier before interpreting any results.

Generate fresh reports against an explicit tier, for example:

```bash
make estimate-flops-frozen-backbone
make report-wiring-budgets
```

Raw checkpoints and telemetry remain ignored under each tier's `results/`
directory. The parameter-free `strided_self_attention` control is not a
Phase-A wiring arm.
