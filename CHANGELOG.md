## [Frozen Candidate Rank Navigation Ablation] - 2026-11-07
### Features
- Add an opt-in policy that selects one candidate at episode start with the existing frozen trajectory rank head, then holds that Goal while using the same guarded path and Stop protocol.
- Add explicit arm selection to the expanded-public navigation evaluator so a single frozen ablation can be run independently and resumed.
### Design Rationale
- The current Val-Seen policy forced candidate slot 0 on every decision, leaving the trained candidate rank head unused. This ablation isolates whether its existing ranking signal can improve navigation without RGB or weight updates.
### Notes & Caveats
- This is a Geometry/controller ablation, not visual evidence. It must not be used as the visual fallback. The Geometry-only ordering remains the fallback when no reliable RGB evidence exists; any gains from this arm are attributed to the learned controller rank head only.

## [Public Landmark Name Variant Retrieval] - 2026-11-07
### Features
- Expand public CityRefer landmark matching with collision-checked unique name phrases, provider-prefix removal, and final-token singular/plural variants.
- Add regression tests for shorthand binding, homonym rejection, and retained Polygon identity.
### Design Rationale
- Improve landmark anchoring from instruction text without using target identity, coordinates, or target-conditioned selection. Keep all Polygons for a repeated canonical public name.
### Notes & Caveats
- TRAIN Cambridge development and Birmingham confirmation improved under frozen Geometry and decoding. The frozen full 2470-task Val-Seen localization evaluation reached 50.6478% (1251/2470) with raw V2 Geometry and the expanded resolver. This is initial localization, not closed-loop navigation SR; differences from the earlier 49.9595% best also include checkpoint/protocol-arm differences and cannot be attributed solely to the resolver.

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

## [Public Landmark Correspondence Review] - 2026-11-07
### Features
- Match public CityRefer named buildings and road junctions to OSM across 34 scenes; fit city similarity proposals with physical-object-isolated holdouts.
- Add authenticated correspondence review, paired polygons, evidence, immutable fit/validation roles and persistent identity decisions.
- Evaluate two frozen V2 Geometry checkpoints on all 2470 HETT Refined Val-Seen tasks with raw and fixed 20m spatial-NMS recall.
### Design Rationale
- Keep identity uncertainty distinct from geographic accuracy; no automatic validation promotion or replacement of existing approved registrations.
### Notes & Caveats
- Automated heldout anchor residuals are proxies. No newly independently confirmed scenes; V2 uses blind public text binding, explicitly differing from historical annotation-reference metadata.

## [V2 Public Binding and Top-1 Training Experiments] - 2026-11-07
### Features
- Align all 21,878 TRAIN queries with blind public-name binding; isolate training labels and physical-target groups.
- Add scoped public-name spelling/abbreviation retrieval and retain literal aliases for existing geometric relation parsing.
- Add TRAIN-only target-region/ranking and source-correct belief preservation losses with frozen-model controls.
### Design Rationale
- Diagnose binding mismatch and forgetting on TRAIN before expanding training; freeze name resolution before full Val measurement.
### Notes & Caveats
- Full2470 FP32 matched Top1 improves 46.96% to48.26%, net+32; the >50% goal remains unachieved. Initial loss-only adaptation failed its TRAIN confirmation; fidelity training is in progress. No Test, Stop changes, OSM model input or main merge.
