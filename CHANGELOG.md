# Changelog

## [direction-frames-v4] - 2026-11-06
### Features
- Add per-relation semantic frame evidence, full-TRAIN explicit-role search, conservative target-to-named-road travel anchors and scope repairs.
- Freeze region-valued observation headings and complete directed road-centerline fields before GT evaluation; validate road topology/tangents across map resolutions.
- Add full-contour road-side checks, unknown/coverage accounting, held-out-map breakdowns, matched baselines, wrong-binding/role-swap/rotation controls and real-map failure cases.
### Design Rationale
- Separate a computable proxy from a uniquely identified scalar heading or directed spatial field; retain ambiguity of broad viewpoints, unsigned axes and missing facades.
- Treat language-to-frame composition as a testable partial mechanism; do not infer author conventions from target-optimal angles or low world-axis agreement.
### Notes & Caveats
- No model training or navigation experiments; TRAIN only. Prior v1/v2/v3 data and all old bound inputs preserved.
- Geometry-model uniqueness does not verify the annotator camera pose. High conditional road-side rates have limited coverage; matched overall-gain confidence intervals still include zero.
- External review status is logged separately from local reproducibility checks.

## [direction-frames-v3] - 2026-11-06
### Features
- Preserve all v2 bindings and expand named-road binding by independent ontology constraints; retain unresolved subjects and reference frames.
- Add GT-free local-frame candidate construction, explicit observation/road-origin rules, frozen BERT silver-label routing, object/map crossfit, matched-support nulls and Chinese report with real-map figures.
### Design Rationale
- Separate a licensed single heading from an unresolved candidate mixture; score fixed mixture mass, never a target-GT per-sample best frame.
- Evaluate signed road-side geometry separately from centroid angular sectors, and compare on matching world-baseline support.
### Notes & Caveats
- Silver-label prediction is not annotator-frame recognition. Known v2 boathouse is marked as development evidence and excluded in a sensitivity cohort.
- Straight-road and water-bank headings are independent geometry proxies; intrinsic facade direction and author camera pose remain unverified.
- No navigation network, BERT finetuning, full navigation training, Val or Test data involved. External review execution status is recorded separately from local checks.

## [direction-frames-v2] - 2026-11-06
### Features
- Restrict the primary population to direction relations whose reference is a named landmark; retain the real relation subject and abstain on unlocalized context objects.
- Re-audit all four candidate contradictions with raw TRAIN descriptions, real map polygons, independently specified stream-facing evidence and explicit limits on author-view recovery.
- Add text-only dependency/surface scope agreement, confidence strata, joint rotations/reflections with grouped crossfit, motion proxies, unsigned local axes and matched confidence intervals.
### Design Rationale
- Treat v1 opposite-word examples as reference-frame audit candidates, not confirmed same-frame annotation contradictions.
- Never choose landmark identities or PCA signs using target GT; compare geometry anchors without changing binding or refitting the dictionary.
### Notes & Caveats
- Source scope confidence is a rule score, not a calibrated probability; unresolved context subjects stay excluded.
- Future trajectory and target-conditioned approach windows are offline diagnostics only. Directed facade/car/parking-lane headings and true geographic north remain unavailable.
- External adversarial-review was actually invoked but failed to return due runtime reconnection errors; preserve logs and distinguish local checks from independent review.


## [direction-frames-v1] - 2026-11-06
### Features
- Independent full original CityNav TRAIN spatial-direction audit with original CityRefer contours.
- Conservative relation scope extraction, offline geometric verification, four reference-frame hypotheses, clustered confidence intervals, high-resolution figures and Chinese report.
### Design Rationale
- Keep source network behavior untouched; isolate exploratory directional statistics from model inference.
- Require unique text-based reference binding; preserve indirect/unknown relations instead of using target GT to resolve them.
- Separate fitted XY concentration from independent evidence about an annotator's reference frame.
### Notes & Caveats
- Exact-name/simple-scope coverage is low; rates do not estimate full-corpus frame frequencies.
- Missing CRS and directed heading make true geographic north and intrinsic orientation unavailable; initial trajectory yaw is not annotation yaw.
