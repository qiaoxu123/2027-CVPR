# AerialVG Candidate-Attention / Reliability-Gated Belief Fusion (experiment)

**Purpose:** test whether a globally competitive visual goal *probability
distribution* can repair failures of the bounded additive residual without
destroying A0 correct states. This commit contains code/tests and an isolated
TRAIN/Dev experiment runner; it contains **no measured improvement**.

## Architecture
1. Freeze the existing AerialVG Native Plain checkpoint (including its query
   ranking and null head). Do not train/recompute AerialVG backbone.
2. Form query attention from frozen query logits plus the null alternative.
   Only query kernels with actual visible grid support are retained.
   Renormalize each supported Gaussian kernel over observed cells.
3. Combine query attention with supported kernels into a normalized visual
   goal distribution P_V on the global 28x28 grid. This is lightweight
   posterior attention, **not** a new or official AerialVG Cross-Attention
   layer.
4. Train a tiny 16-feature MLP to estimate if switching from frozen A0 P_0
   to a visual-driven candidate is likely to *correct*, rather than harm,
   the top-1 goal. Features: null probability, supported mass, prior entropy
   and margin, query-score gap, visual-vs-prior agreement, peak distance,
   quality, observability, etc.
5. If disabled, no supported evidence, or reliability < chosen threshold,
   return exactly the original A0 logits. Else perform
       P_final = (1 - transfer) * P_A0 + transfer * P_V
   with a fixed transfer of 0.9 (configurable for a new experiment). This
   permits the visual peak to replace a confident wrong A0 peak, without
   increasing the old residual bound. The final logit normalization is
   preserved. Target GT is NEVER a forward argument.

## Training labels and protocol
- The new gate label is TRAIN-only: A0 grid argmax >20m from target AND
  *the actual fixed-transfer visual/global mixture* would make argmax <=20m.
  Do not substitute "some query within20m" for a recoverable action.
- These are spatial utility labels, **not verified object-instance labels**.
  They do not prove that the visual object is the target.
- Grouped TRAIN split by SHA256(group) into fit and disjoint calibration,
  80:20 approximately. No independent DEV can choose epoch, threshold,
  transfer, source model, or architecture after viewing its performance.
- Final gate after fixed epochs; threshold chosen exclusively by TRAIN
  calibration to maximize corrected minus damaged, constraining calibration
  damage to <=2% of rows by default. A threshold of 1.0 means no-op.
  No Dev-derived refitting or hyperparameter search.
- The original 1275 TRAIN / 635 internal Dev caches are reused.
  All existing manifest file hashes, base SHA, group partitions and frozen
  Native checkpoint SHA are checked. No formal Val-Seen/Unseen/Test access.
- The emitted checkpoint has `admitted_for_runtime=False` and there is
  intentionally **no production installation hook**. The historical
  paused full Seen run and original code paths are untouched.

## Reproduce locally
In the repository checkout after switching to the new branch:

```bash
python -m pytest -q tests/test_native_gated_belief_fusion.py

python -m scripts.train_native_gated_fusion \
  --train Ablations/GraphNav/visual-residual-20261013/aerialvg_native_training_train256_20261101 \
  --dev Ablations/GraphNav/visual-residual-20261013/aerialvg_native_training_dev128_20261101 \
  --native-checkpoint Ablations/GraphNav/visual-residual-20261013/aerialvg_native_plain_sigma8_s0_20261101/best.pt \
  --output Ablations/GraphNav/visual-residual-20261013/native_gated_fusion_seed0_20261101 \
  --epochs 6 --batch-size 16 --seed 0 --device cuda
```

Run on one idle GPU; output path must not already exist. CPU synthetic unit
tests do not need the AerialVG model files. Full training needs local
existing frozen checkpoint/caches. Run seed1 and seed2 in NEW output
directories only after seed0 passes basic controls.

## Required report
- Match the existing 635 Dev states and original A0 / frozen residual
  counts (A0 correct 437/635, residual 444/635: corrected 22, damaged 15).
  Treat any mismatch as a provenance/runtime compatibility blocker.
- On TRAIN calibration: label positives, threshold, accepted, corrected,
  damaged, no-op selection, grouped feature distribution.
- On internal Dev: A0/residual/new gate R@1 and NE, corrected and damaged,
  acceptance rate, eligible switch fraction. Report error-category overlap;
  do not add overlapping Oracle counts.
- The earlier 37 failed cases are label-derived diagnostics, not a held-out
  result or proof of instance identity. DO NOT tune on these 37 Dev states.
- If net benefit does not exceed residual +7/635, or harm rises, report a
  negative result and do not enable native gated fusion for navigation.
- If a meaningful and independent positive result emerges, review separate
  *frozen* Val-Seen navigation experiment eligibility; do not resume or
  alter existing paused runs automatically.

## Known limitations
The visual hypothesis is currently a mixture of observed Gaussian kernels
centred at native query projections, still subject to orthographic geometry,
kernel discretization, and visible support masks. A gate cannot invent a
correct candidate outside the image or eliminate false physical associations.
The 2% TRAIN-calibration damage budget is not an out-of-distribution safety
guarantee. Inference time and closed-loop SR are *not* yet measured.
