# Native-preserving relation Gate v2

Base b83287e17e34542ae4b1a8d407e1859149987c2e; independent experiment branch.

1. Audit TRAIN only before design freeze: enumerate named entities/parsed and ambiguous relations, coverage waterfall and real multi-anchor counts. No Dev outcomes.
2. Add separate v2 modules, preserving v1 APIs/checkpoints: split between anchors, real public distractor landmarks selected near named reference centers without GT. Keep name-binding feature explicit; no fabricated target instance labels. Compound directions use existing fixed parser.
3. Add public geometry auxiliary regression and public binding attention loss, candidate 20m weak ranking, final Belief localization loss and KL/GT-correct Native preservation. Freeze A0/backbone/Native.
4. New Gate: proposal = 0.1 Native probability + 0.9 relation visual probability; reject/disabled/no relation => bitwise Native Residual. TRAIN labels compare proposal against Native (R@1 first; NE improvement >=1m for tied correctness). Frozen old Gate evaluated on same new relation scores as control.
5. TRAIN fit 203 groups; existing calibration49 groups split again by group, stratified by public named-reference availability, for probability calibration vs threshold selection. Fit monotonic Platt probability calibration only there; threshold grid includes calibrated quantiles <0.5 and exact no-op. Select net gain vs Native under ZERO additional Native-correct damage and ZERO additional A0-correct damage on threshold-selection TRAIN groups. No guarantee of zero damage on Dev.
6. Unit tests then seed0 1+1 smoke with TRAIN-only diagnostics; if finite gradients and strict fallback contracts hold, fixed 4 relation +3 Gate epochs Pilot. No epoch/hyperparameter search; internal Dev read once after frozen fit/calibration/selection. No official validation, Test, closed loop, or paused job changes.
7. Same635 Dev report: A0, Native, direct relation residual, old Gate and new Gate; relation-off and query-geometry-shuffle; report all accepts/harm vs both A0/Native, public hard-negative vs genuine multi-positive counts, Q/K gradient and attention selection. Preserve negative results; commit/push code and evidence.

Implementation calls are agent choices, fixed before training, not user-chosen hyperparameters. Read-only existing public map cache supplies references; only TRAIN labels enter training losses/calibration. No reference is chosen using target coordinates. Hard-negative name binding is a public correspondence, not a verified target instance label.

Smoke correction before Pilot: a pure hash split put every eligible calibration group on the probability side, leaving threshold selection without eligible samples. Replaced by deterministic public-availability stratification (no GT or outcome stratification), and repeat TRAIN-only smoke in a new directory. Preserve the initial smoke evidence; no internal Dev has been read.

TRAIN-only Smoke found the probability subset has zero positive Gate labels (10 eligible states). Fixed calibration admission rule before Pilot: if either class is absent, still record low-threshold search but force exact Native no-op; near-constant Laplace fallback scores are not evidence of calibrated discrimination. Full Pilot remains fixed 4+3, with no Dev-based changes.
