# Changelog

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
