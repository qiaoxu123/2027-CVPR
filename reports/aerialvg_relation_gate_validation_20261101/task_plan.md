# RG-grounding validation

Base branch research/aerialvg-relation-gated-grounding-20261101, exact a14292c1677964e358b278c7e948f0abfecf929a.
1. [done] Run requested tests and verify parser/world coordinates/masks/gradient/fallback/no-GT contracts on actual Train data; fix concrete failures only.
2. [done] seed0 smoke:1 relation epoch +1 gate epoch. No backbone training or formal validation/Test data.
3. [done] After smoke passes, seed0 full:4 relation epochs +3 gate epochs; threshold fit/calibrated on Train groups only.
4. [done] Same635 internalDev comparison A0/Native/Relation/Relation+Gate, relation-off and shuffled-descriptor ablations, named-relation/supervised counts and gradients; Chinese report.
5. [done] Verify fixes/results and push commit; preserve original formal experiments and frozen checkpoints.
