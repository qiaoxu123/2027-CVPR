## [UnifiedPGVB-V3 training runtime] - 2026-11-06
### Features
- Add source-checked V3 warm start, mid-epoch optimizer/RNG checkpoints, cumulative training time budget and TRAIN-loss plateau stopping.
- Reuse frozen dense RGB features only for identical image SHA; support training without automatic subset evaluation.
### Design Rationale
- Bound training to four hours and preserve resumability; full Val-Seen evaluation follows frozen training rather than repeated subset selection.
### Notes & Caveats
- Pilot localization is not navigation SR. Initial V2 weights have seen TRAIN. Raster height is a depth proxy; map-guided planned views differ from causal navigation observations.

## [UnifiedPGVB-V2 paper identity and results documentation] - 2026-11-06

### Features
- Standardize UnifiedPGVB-V2 as the paper's primary method; retain GraphNav-V1 as historical baseline.
- Record verified full Val-Seen results and matched RGB ablation, with immutable experiment references.

### Design Rationale
- Keep method identity, experimental version and measured performance distinct; remove stale no-results claims.

### Notes & Caveats
- Documentation update only; no experiment code/weights merged or evaluation rerun. Unseen results remain unverified for this experiment.


## [OSM Control Point Annotation] - 2026-11-07
### Features
- Add authenticated dual-map public control-point annotation, atomic city-level persistence and independent RANSAC validation.
### Design Rationale
- Keep local XY visualization available before georeferencing; preserve shared city transform and per-scene validation rather than assume 34 independent alignments.
### Notes & Caveats
- OSM registration requires real human controls; current public deployment uses a working fallback FRPS while requested host credentials remain unavailable. No model weights or inference pipeline changed.

## [OSM Structural Auto Registration Pilot] - 2026-11-07
### Features
- Add fit-only public-control initialization, OSM vector retrieval, RGB/height structural proxies, automatic centroid RANSAC and robust boundary refinement.
- Add automatic proposal, overlay/structural heatmap and explicit accept/reject UI with source-revision checks.
### Design Rationale
- Select predictions only by public structural costs, then evaluate existing independent controls; preserve manual controls and avoid assuming city-wide validity from one scene.
### Notes & Caveats
- Birmingham block3 improved on five independent controls; Cambridge has no independent validation. Masks/junctions are geometric proxies; no claim of validated alignment across all 34 scenes.

## [CityNav OSM 34-Scene Registration Audit] - 2026-11-07
### Features
- Run resumable city-seeded registration on all 34 public scenes with independent per-scene grading and a 34-scene HTTP dashboard.
- Add scene-specific accept/reject, control fingerprint checks and validated-only OSM geometry/semantics export.
### Design Rationale
- Freeze city/global-vs-local selection using structural evidence, then evaluate independent controls only in their own scene. Never turn a structural score into geographic accuracy.
### Notes & Caveats
- 34 computational outputs: 1 independently verified, 21 awaiting validation, 12 unreliable local-correction cases. No scene proven within 2m maximum error; no claim of city-wide geographic validity and no model training.
