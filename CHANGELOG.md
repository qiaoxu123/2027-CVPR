# Changelog

## [0.2.0-a2-dev] - 2026-10-28
### Features
- Add target-instance-disjoint A2 supervision protocol and full CityNav train source audit.
- Validate public landmark identities and complete BERT mention spans; retain ambiguity and truncation as unknown.
- Integrate observed native384 WALDO boxes, actual frozen SigLIP2 box crops, per-landmark BERT context and conservative updates through an optional frozen-A0 hook.
- Add candidate trust supervision, variable-shape batching, causal resumable extraction and content hashes for annotations, original dependencies and all TRAIN rasters.
- Audit coarse Car click regions with authoritative positions and symmetric unique association; ambiguous and cross-partition boxes remain unknown.
### Design Rationale
- Extend supervision beyond the old 200-instance pilot without moving benchmark labels into runtime inputs.
- Preserve frozen A0/A1 and use an isolated optional A2 integration.
- CityRefer Car circles do not represent vehicle segmentations. Position-based associations require a single contained annotation point, <=1m detection-centre distance, >=1m margins in both directions, confidence >=.5 and a consistent containing annotation region.
### Notes & Caveats
- 300 CPU tests, real frozen-A0 CUDA parity and informative TRAIN ranking backward gates passed. Full cache, full 20+5 training, independent calibration and full navigation evaluation are in progress. No A2 performance claim is available.
