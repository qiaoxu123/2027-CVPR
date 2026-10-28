# A2 next revision: test gain/harm discrimination before full retraining

Goal remains actual A2>A0 on BOTH frozen full Seen2470 and Unseen2697, not cached metrics. No benchmark opened for revision selection. Preserve all v1 code, weights, calibration and originalA0/A1.

## New verified evidence
- v1 independently collected256 unused TRAIN-source targets/2033 real frozen-A0 observations; FIT127/1004, VERIFY129/1029. Calibration20.23s. Unit exited0; GPU empty.
- Unique FIT choice scale.1 threshold.65 T1 gives VERIFY3gain groups/0broken; R1 +.60237pp CI[0,1.46985]pp, NE−.14007m. Predeclared minimum5gain and strictlypositiveCI lower fail. Correctly OFF, no pilot/formal evaluation launched. Independent CPU source+packet audit and unique-choice verification reproduce result (NErounding3.6e-9m only).
- FIT127 only: unscaled trained gate corrects23views/10groups but breaks33views/20groups. GT oracle over4fixed mixture weights can recover26views/12groups. Benefit probabilities mean .6489 for gains vs .6355 for harms; gain-vs-harm AUC .54545. Expected-distance improvement is a poor discriminator of actual20m peak outcomes. This motivates testing direct outcome/risk prediction, not lowering acceptance criteria or picking a VERIFY runner-up.

## Next authorized steps (TRAIN evidence only)
1. Extract frozen v1 public proposal/A0 context and explicit gain/harm labels on the original31415TRAIN/4618dev views. Keep all GT strictly in offline labels. Pin checkpoint/source/code and require complete row counts.
2. Test whether public joint A0/proposal confidence, entropy, peak distance, prior/proposal agreement and observed match evidence can discriminate actual corrected/broken states. Use original target-disjoint TRAIN/dev, never v1 VERIFY to choose parameters. This is a diagnostic probe, not a substitute for full20+5 training.
3. If justified, implement versioned new relation/belief outcome heads with explicit correction/harm supervision and public proposal context; fresh full20+5 seed0 with old foundation/controller/grid frozen. Compare alternatives on TRAIN dev only. New modules/worktree as needed: v1 pinned files MUST remain unchanged.
4. Reserve new never-observed targets before any new labels/training; exclude v1all256, old40260cached rows and all train/dev identities. Previous opened v1 confirmation is development evidence, never untouched calibration again. Maintain adequate unseen independent calibration population; do not reuse v1VERIFY as fresh.
5. New independent calibration uses the existing risk standards, then real paired pilot, immutable freeze, all four full A0/A2 jobs serially. No forced gate, no OFF-parity performance claim, no Unseen tuning.

## Remaining deliverables
All full metrics/maps/paired uncertainty/coverage/latency/fallback/curves; genuine valid-target counterfactual-language control (audit of31415TRAINcacheviews found0validdistinct-target same-observation pairs); complete A0/A1 summary already preserved. Current365CPUtests pass and full closed-loop/results implementation reviewed but no new A0/A2 full result exists yet.
