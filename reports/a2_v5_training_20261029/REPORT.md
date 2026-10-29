# A2 V5 Target-Balanced Training — Final Diagnostic Report

- Run: `a2_waldo_landmark_20261027/revision_v5`
- Training directory: `/mnt/hdd8t/Workspace/2027-CVPR-A2-20261028/legacy_home/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBeliefV4/runs/a2_waldo_landmark_20261027/revision_v5/training`
- Report generated: 2026-10-29T14:35:27+08:00
- Training status: **complete** (25/25 epochs; 20 ranking + 5 refinement)
- Completion artifact status: `v5_full20plus5_complete_artifact_verification_pending`
- Selected refinement checkpoint: **epoch 24**
- `checkpoint_eligible`: **false**
- `independent_confirmation_pending`: **true**
- `benchmark_used = false`; `confirmation_used = false`

## Executive summary

- A0 internal DEV baseline: R@1 **0.6804**, mean localization error **19.76 m**.
- Final A2 (epoch 25) internal DEV: R@1 **0.5187**, NE **35.85 m** (Δ R@1 **-16.17 pp**, Δ NE **+16.09 m**).
- Selected A2 checkpoint (epoch 24): R@1 **0.5114**, NE **35.97 m** (Δ R@1 **-16.89 pp**, Δ NE **+16.21 m**).
- Real subset (1029 views) at epoch 25: R@1 **0.3819** vs A0 **0.6531** (Δ **-27.11 pp**); NE **55.04 m** vs **19.08 m** (Δ **+35.96 m**).
- Eligible epochs: **0/25**; epochs with a selected operating point: **0/25**.

**Assessment: negative on internal DEV.** The A2 map output remains substantially below A0 on recall and mean error at every checkpoint. Refinement reduced NE slightly relative to the rank checkpoint but did not recover recall. Because no operating point was eligible in any epoch, the run produced a diagnostic negative result only; it must not be promoted to independent confirmation or benchmark evaluation.

## Final and selected-checkpoint metrics

| Metric | A0 | A2 epoch 25 | Δ | A2 selected epoch 24 | Δ |
|---|---:|---:|---:|---:|---:|
| Overall R@1 | 0.6804 | 0.5187 | -16.17 pp | 0.5114 | -16.89 pp |
| Overall NE (m) | 19.76 | 35.85 | +16.09 | 35.97 | +16.21 |
| Real R@1 | 0.6531 | 0.3819 | -27.11 pp | 0.3810 | -27.21 pp |
| Real NE (m) | 19.08 | 55.04 | +35.96 | 54.78 | +35.70 |
| Teacher R@1 | 0.6864 | 0.5492 | -13.73 pp | 0.5405 | -14.60 pp |
| Teacher NE (m) | 19.91 | 31.57 | +11.66 | 31.77 | +11.86 |
| Proposal R@1 | — | 0.3363 | — | 0.3342 | — |
| Proposal NE (m) | — | 43.12 | — | 43.16 | — |

## Refinement-stage trend (epochs 21–25)

| Epoch | A2 R@1 | A2 NE (m) | Real R@1 | Real NE (m) | Proposal R@1 | Train loss | Eligible |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 21 | 0.5155 | 36.13 | 0.3741 | 55.63 | 0.3303 | 4.794 | no |
| 22 | 0.5201 | 36.21 | 0.3829 | 55.33 | 0.3343 | 4.610 | no |
| 23 | 0.5125 | 36.12 | 0.3810 | 55.19 | 0.3301 | 4.561 | no |
| 24 | 0.5114 | 35.97 | 0.3810 | 54.78 | 0.3342 | 4.542 | no |
| 25 | 0.5187 | 35.85 | 0.3819 | 55.04 | 0.3363 | 4.520 | no |

## Full epoch table

| Epoch | Stage | A2 R@1 | A2 NE (m) | Real R@1 | Real NE (m) | Proposal R@1 | Train loss | Eligible |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | ranking | 0.6037 | 27.63 | 0.5053 | 37.90 | 0.3304 | 3.151 | no |
| 2 | ranking | 0.5860 | 29.89 | 0.4674 | 43.95 | 0.3285 | 2.873 | no |
| 3 | ranking | 0.5800 | 30.52 | 0.4606 | 44.64 | 0.3276 | 2.750 | no |
| 4 | ranking | 0.5601 | 32.23 | 0.4266 | 47.82 | 0.3283 | 2.643 | no |
| 5 | ranking | 0.5647 | 31.65 | 0.4383 | 47.28 | 0.3319 | 2.564 | no |
| 6 | ranking | 0.5596 | 32.53 | 0.4568 | 48.07 | 0.3359 | 2.505 | no |
| 7 | ranking | 0.5523 | 33.57 | 0.4208 | 51.21 | 0.3281 | 2.445 | no |
| 8 | ranking | 0.5484 | 33.84 | 0.4237 | 51.28 | 0.3248 | 2.401 | no |
| 9 | ranking | 0.5502 | 33.93 | 0.4179 | 52.44 | 0.3294 | 2.381 | no |
| 10 | ranking | 0.5412 | 34.95 | 0.4266 | 53.69 | 0.3373 | 2.334 | no |
| 11 | ranking | 0.5424 | 34.74 | 0.4218 | 53.22 | 0.3352 | 2.310 | no |
| 12 | ranking | 0.5387 | 35.27 | 0.4043 | 55.55 | 0.3338 | 2.289 | no |
| 13 | ranking | 0.5380 | 35.69 | 0.4130 | 56.32 | 0.3393 | 2.258 | no |
| 14 | ranking | 0.5305 | 37.24 | 0.4023 | 58.35 | 0.3308 | 2.248 | no |
| 15 | ranking | 0.5274 | 36.61 | 0.4062 | 58.75 | 0.3333 | 2.211 | no |
| 16 | ranking | 0.5265 | 37.19 | 0.3916 | 58.88 | 0.3345 | 2.194 | no |
| 17 | ranking | 0.5254 | 36.94 | 0.4023 | 57.50 | 0.3345 | 2.184 | no |
| 18 | ranking | 0.5178 | 37.60 | 0.3887 | 58.68 | 0.3311 | 2.174 | no |
| 19 | ranking | 0.5265 | 37.06 | 0.4062 | 57.96 | 0.3386 | 2.160 | no |
| 20 | ranking | 0.5224 | 37.59 | 0.3897 | 58.96 | 0.3382 | 2.151 | no |
| 21 | refinement | 0.5155 | 36.13 | 0.3741 | 55.63 | 0.3303 | 4.794 | no |
| 22 | refinement | 0.5201 | 36.21 | 0.3829 | 55.33 | 0.3343 | 4.610 | no |
| 23 | refinement | 0.5125 | 36.12 | 0.3810 | 55.19 | 0.3301 | 4.561 | no |
| 24 | refinement | 0.5114 | 35.97 | 0.3810 | 54.78 | 0.3342 | 4.542 | no |
| 25 | refinement | 0.5187 | 35.85 | 0.3819 | 55.04 | 0.3363 | 4.520 | no |

## Train / DEV gap at epoch 25

- Train map R@1: **0.6696** vs DEV map R@1 **0.5187** (gap **0.1509**).
- Train map NE: **28.91 m** vs DEV map NE **35.85 m** (gap **-6.93 m**).
- Train proposal R@1: **0.5336** vs DEV proposal R@1 **0.3363** (gap **0.1973**).

## Gate and provenance

- Best ranking checkpoint by the code selection key: **epoch 17** (real proposal R@1 0.2342, real proposal NE 75.92 m).
- Gate: **0/25** epochs eligible; **0/25** epochs with a selected operating point.
- `training_complete.json`: selected epoch **24**, `checkpoint_eligible=false`, `independent_confirmation_pending=true`.
- No independent confirmation artifacts were found under `revision_v5`; the completion artifact explicitly marks confirmation as pending.
- `protocol.json` sha256: `e52297681010e0b626d0f062fd863b32de042eadd24b6bcc41cffcb98861f5d0`
- `history.json` sha256: `152ab0df5490721c3eb9506acc7ac367d42fbbdad81bb84995c8eccca5c6f8f3`
- `training_complete.json` sha256: `6a1dd317f591a90875362b0e55f582497ab499a9ac0d52ad3fbb7531d657e744`
- Source reference: `origin/research/a2-waldo-landmark-belief-20261027` at commit `6161547`; key v5 source blobs match this commit, and the exact per-file hashes are recorded in `protocol.json`.

## Recommendation

Do not enable independent confirmation, FIT collection, or benchmark evaluation for this checkpoint. Retain the run as a diagnostic negative result. The refinement stage slightly reduced mean localization error but lost recall and never approached A0; more epochs of this recipe are not expected to change the conclusion. A further attempt should change the fusion/gating objective or supervision rather than extend training.

