# Changelog

## [0.2.0-a2-dev] - 2026-10-28
### Features
- Add target-instance-disjoint A2 supervision protocol and full CityNav train source audit.
- Validate public landmark identities and complete BERT mention spans; retain ambiguity and truncation as unknown.
- Integrate observed native384 WALDO boxes, actual frozen SigLIP2 box crops, per-landmark BERT context and conservative updates through an optional frozen-A0 hook.
- Add candidate trust supervision, variable-shape batching, causal resumable extraction and content hashes for annotations, original dependencies and all TRAIN rasters.
- Audit coarse Car click regions with authoritative positions and symmetric unique association; ambiguous and cross-partition boxes remain unknown.
- Add a fixed full20+5 trainer with all-view epoch checks, TRAIN-dev selection, per-epoch checkpoints, TensorBoard and crash-safe history/best-model recovery.
- Condition candidate trust and bounded offsets on the same visual/language/geometry pair context used by ranking.
- Add a code-pinned CPU waiter that inherits the GPU lock into sequential CUDA-preflight and training children.
- Add complete training artifact/optimizer/curve verification and independent target-group fit/verify calibration with conservative fallback.
### Design Rationale
- Extend supervision beyond the old 200-instance pilot without moving benchmark labels into runtime inputs.
- Preserve frozen A0/A1 and use an isolated optional A2 integration.
- CityRefer Car circles do not represent vehicle segmentations. Position-based associations require a single contained annotation point, <=1m detection-centre distance, >=1m margins in both directions, confidence >=.5 and a consistent containing annotation region.
### Notes & Caveats
- 318 CPU tests passed. Real frozen-A0 CUDA parity and the initial prototype's informative TRAIN backward check passed; the updated geometry-conditioned trust/offset model must pass a new real CUDA gate after full-cache completion.
- Full cache, full20+5 training, independent calibration and full navigation evaluation remain incomplete. No A2 performance claim is available.
