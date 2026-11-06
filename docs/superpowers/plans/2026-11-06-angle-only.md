# PR25 Angle-only experiment execution

Goal: train only eight-bin bearing and bounded fusion, with frozen existing full Unified V2. All21878 TRAIN, seed42,3epochs; full2470 Val-Seen predicted-only navigation.

- [x] CPU review and tests: angular scale invariance, bin convention, mask/padding, zero-update, nonzero head gradients, no GT forward, candidate/XY/context alignment.
- [x] Fix target-indexed source reference lookup in angle experiment: blind public catalog binding identical across TRAIN/runtime and matched existing public-bound full-text baseline. Preserve original full checkpoint and original source inventory SHA; document binding distribution shift.
- [x] Fix opt-in runtime selection/capture; no Oracle runtime mode, no threshold changes.
- [x] Real CUDA Smoke12updates, then3full epochs with cached frozen BERT and resource/loss/gradient/checkpoint logs.
- [x] TRAIN-only predicted/Oracle angle localization and CF diagnostics. Freeze epoch3, then all2470 Val-Seen, including failures/no anchors. No Val selection or Test.
- [x] Report and real Polygon/angle/Belief visuals, tests, adversarial review invocation, commit/push PR25 without main merge.

New head input whitelist: full frozen text/name embeddings, candidate map angle, named anchor validity, frozen baseline logits. No dx/dy, radial distances, range buckets, GT position/ID, future observations or Oracle angle. Existing frozen model retains its original geometry features; this is not a geometry-free model. Map+X is0deg, +Y90deg, no geographic north claim.
