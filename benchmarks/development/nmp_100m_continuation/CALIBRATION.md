# Mixed-pass NMP calibration

Calibration ran on 2026-08-25 on one NVIDIA RTX A6000 with driver 580.126.09.
It used the exact 100,007,936-token hybrid snapshot, shared-final targets, the
90% K=2 / 10% K=3 mixture, eight fixed data blocks, and 32 predictor-only
warm-up blocks at predictor LR `3e-5`.

## Provenance

- Source snapshot SHA-256:
  `26e1924ebbf8c4fc6c2501ef861227639c21bb5f702ff914b4ed117f1822c7d0`
- Continuation data-manifest SHA-256:
  `2116ce601a79dda93a6e02d20c9435898d00909db33d5204cda552ae7805b6a1`
- Calibration source commit: `b36505488b5d894e577994a3df57dc003e33563e`

## Exact 90/10 mixture after head warm-up

| Measurement | Value |
| --- | ---: |
| NTP pretrained gradient norm | 1.0659523532 |
| Unit-weight Bank NMP pretrained gradient norm | 0.1245950513 |
| NTP Memory Attention/Recirculation gradient norm | 0.3658976573 |
| Unit-weight Bank NMP memory gradient norm | 0.0578942659 |
| Bank NMP versus NTP pretrained-gradient cosine | -0.0421926121 |
| Bank NMP versus NTP memory-gradient cosine | -0.0661537419 |
| Initial Bank NMP loss | 0.6604691394 |
| Post-warm-up Bank NMP loss | 0.5793813281 |

The coefficient targeting 5% of the pretrained NTP gradient norm is:

```text
0.05 * 1.0659523532 / 0.1245950513 = 0.427766729982766
```

The corresponding 10% and 20% candidates are `0.855533459965532` and
`1.711066919931064`. This campaign uses the 5% value. The previous provisional
`0.8510068634` coefficient is therefore approximately the 10% setting on this
parent and is not used.
