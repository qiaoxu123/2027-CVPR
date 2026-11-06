# Changelog

## [sentence-parser-cache-v1-in-progress] - 2026-11-06
### Features
- Retain semantically incomplete batches as rejected history, return them to pending, and count restored records only after full re-parsing.
- Publish deterministic gzip JSONL with compressed/uncompressed SHA256; retain generated plain JSONL and SQLite locally.
- Export all CityNav split texts with exact-text deduplication and per-episode lookup aliases.
- Start current-assistant-authored offline semantic graphs; retain relation scope, role alternatives, view evidence and unknown frames.
- Add validated JSONL serialization and lookup-only SQLite cache with explicit missing-record errors.
### Design Rationale
- Follow the requested direct model parsing workflow without CLI/API parsing or heuristic substitute labels.
- Keep text parsing separate from target GT, geometric binding, and v1–v5 scientific evaluation.
### Notes & Caveats
- Partial cache; live counts and model composition are in parsed/progress.json, with no full-dataset performance claim.
- User changed subsequent parsing sessions from gpt-6.1-sol low to available gpt-6-luna low; preserve each record's original model attribution.
- All split texts are authorized for parsing; no target coordinates, future trajectories or navigation model training used.
- Requested model/effort recorded as requested configuration, without claiming independent runtime attestation.

## [direction-frames-v5] - 2026-11-06
### Features
- Pin the complete v4 experiment and reproduce its 44 directed-road cases; audit Abbey conflict and classify all 22 ambiguous contours without GT-based sign changes.
- Add full-TRAIN text-role retrieval and conservative new syntax/scope checks, with explicit new-expression, missed-rule, near-duplicate and development-map/road/object strata.
- Compare frozen local fields, world axes, start bearing, yaw and unsigned PCA on identical contour projections/support; isolate exploratory geometry smoothing and retain failed controls.
### Design Rationale
- Keep geometry parameters frozen while testing language adaptation; a newly parsed role is not evidence that the original end-to-end parser generalized.
- Separate retrospective evidence from blind external validation, and conditional consistency from overall coverage or true author-frame recognition.
### Notes & Caveats
- TRAIN only; no Val/Test, graph/navigation/GeomBelief training or navigation SR experiments. No existing model or v4 artifact changed.
- New geometry-scorable evidence remains small and includes repeated development objects; no new-map unique-field success is established.
- Public contour and strict-road-side ambiguity persists; exploratory smoothing has no independent lane-centerline ground truth. External review execution status is logged separately.

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
