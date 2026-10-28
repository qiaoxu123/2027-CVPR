# A2 Road Reference Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore justified public road references without changing legacy A0/A1/V3 or claiming unverified A2 improvement.

**Architecture:** New modules own exact public regions and versioned reference resolution. Existing runtime remains opt-in: new cache reconstruction and live perception share the same resolver, geometry and explicit missing grouped-height policy. Train/evaluate a new version only after complete audits and serial GPU admission.

**Tech Stack:** Python3.10, Shapely, NumPy, frozen BERT/PyTorch, existing verified source manifests.

**Spec:** docs/superpowers/specs/2026-10-29-a2-public-road-reference-design.md

## Global Constraints

- Preserve original A0/A1 results and running remote V3 source/cache bytes.
- Frozen A0/WALDO/BERT/SigLIP2/controller; seed0, 28x28 grid, target-disjoint roles.
- Exact normalized public names only; no private target data in name resolution.
- New road groups require valid connected TrafficRoad/Footpath public polygons including TrafficRoad.
- Grouped height is missing in BOTH training reconstruction and inference; singleton path unchanged.
- Full 20+5 training; independent calibration; final complete Seen2470/Unseen2697; no Unseen tuning.

## Review Focus

- Point-touching members can produce MultiPolygon, not a single exterior.
- Holes/concavities must remain outside rather than become filled target support.
- Duplicate or contradictory source IDs must not silently substitute geometry.
- Alias collisions and repeated query spellings must not add duplicate evidence.
- A previously masked reference has no cached BERT features; flag flipping is insufficient.

### Task1: Exact public region primitive

Create experiments/a2_waldo_landmark_belief/public_regions.py and tests/test_a2_public_regions.py.

Interfaces: region_from_contours(contours: Sequence) -> dict with version/geometry; region_shape(region: Mapping) -> Shapely area geometry; region_statistics(region: Mapping, point: Sequence) -> dict(area,center,signed_distance,inside). Consume only public coordinate rings. JSON-compatible geometry retains holes and components, version public-region-v1. Reject malformed/invalid/nonfinite/nonpositive members before union; never repair with buffer or hull.

- [x] Write tests for overlap union area/centroid, concavity gap, hole, point-touch components, invalid/nonfinite/empty member, JSON roundtrip and order invariance.
- [x] Run CPU pytest; verify missing-module failure.
- [x] Implement primitive with exact unary_union and boundary distance, covers for boundary inclusion.
- [x] Run focused CPU tests; record results. Commit tested primitive as inactive infrastructure.

### Task2: Versioned road reference resolver

Create public_road_references.py and tests/test_a2_public_road_references.py. Interface resolve_road_references(map_name: str, query_names: Sequence[str], named_objects: Sequence[Mapping]) -> list[dict]. Reuse legacy formatting normalization; preserve singleton semantics, conservative unresolved records, deterministic entity IDs/membership and canonical names. Group only complete exact-name match sets meeting the spec. No default switch in policy.py yet.

- [x] Write failing tests for touching/disconnected road names, building/footpath-only rejection, repeated query deduplication, malformed geometry, contradictory duplicate IDs and permutation invariance. Actual source resolver supports canonical public names only, so no speculative new alias source is introduced; existing overlap masking remains the authority.
- [x] Run failing tests, implement resolver using Task1, run focused plus original data_protocol/mention suites.
- [x] Run a complete public TRAIN/DEV reference inventory without loading confirmation or benchmark data; report grouping reachability by observations AND distinct target identities.

### Task3: Shared new-version feature adapter and cache reconstruction

Create versioned region-pair geometry and feature reconstruction modules/tests; do not modify frozen V3 defaults. Consume Task1/2 outputs, original checked candidate/public tensors, annotation-matched instruction, original source/roles signatures. Produce exact union relationship features, regenerated frozen BERT name features, mentions, new labels using same union centroid and new provenance manifests. Group height fields=0/invalid in both offline/live paths. Clone tensors before writes.

- [x] First address the17 source instructions/56 TRAIN observations lost by legacy whole-reference overlap masking: design and test occurrence-specific public mention handling in a new module; keep legacy matcher unchanged. Test separately mentioned road plus nested building name, exact shared-name ambiguity, partial crossing spans and truncation. Repeat complete inventory before feature building; do not infer identity from private labels.
- [x] Add shared reference_geometry.py centroid and twelve-slot relation functions; verify singleton byte parity, exact union area/center/boundary, hole exclusion, missing group height and aligned empty axes. Not yet integrated with cache rebuilding or live perception.
- [ ] Write failing tests for union geometry features, label centroid agreement, singleton legacy parity, grouped height offline/live equality and parent immutability.
- [x] Implement public-only reference_rebuild.rebuild_reference_pack(pack, meta, *, instruction, offsets, references, encode_names) and private reference_training.build_reference_training_row(raw, *, target_kind, real). Tests cover actual feature regeneration rather than flag flipping, cloning immutable inputs, missing new-reference heights and singleton supervision parity. Future live wrapper must call the same public adapter after legacy perception, so newly available singleton heights remain missing in both paths.
- [ ] Implement shared adapter then run tests; audit all39,087 source rows for required reconstruction fields and exact roles.
- [ ] Regenerate new version caches, reuse immutable image features, verify parent/new hashes and no forbidden records.

### Task4: Admission and complete experiment

- [ ] Review actual code using required adversarial-review skill; if unavailable report inability, do not claim approval. Resolve findings and run appropriate CPU tests.
- [ ] After current job is terminal, deploy separately pinned new version, run actual CUDA backward and A0/OFF parity, then full20+5 training serially.
- [ ] Independently calibrate/freeze admitted checkpoint; preserve failed candidates honestly.
- [ ] Complete required paired closed-loop Seen2470/Unseen2697 with metrics, map breakdown, paired uncertainty, logs/curves/checkpoints. No goal completion based on software tests or mention recovery alone.

## Execution ledger

- 2026-10-29: User's persistent implementation/iteration request authorizes native execution; no additional execution-method question. Independent builder checked design and actual old source schemas. Only CPU development while remote V3 continues. No production repair enabled yet.
- Task1/2: observed missing-module failures before implementing both modules; malformed serialization tests then exposed four validation failures, fixed with explicit geometry schema checks. All42 focused+legacy tests pass, including the explicit conservative overlap fallback. Inventory covers all33,440 TRAIN and5,647 DEV observations, restoring at least one reference in6,126/1,124 observations over687/121 targets. TRAIN loses56 old reference observations; classification audit must precede deployment. Software/data checks do not prove model improvement.
- Ruling: use existing exact canonical public-name source only, not speculative new alias fields — matches actual resolver input contract and prevents invented linkage. No current runtime consumer imports the new modules.
- Required adversarial-review skill not found in installed filesystem/tool discovery; no code-review approval is claimed. Keep new modules undeployed pending that check and remaining feature/cache work.
- Additional audit recorded all17 lost-reference source instructions: nested public names (Castle Street/12 Castle Street etc.) trigger legacy whole-reference overlap masking even where separate literal road mentions exist. This changes next action: occurrence-specific mention handling must be resolved before Task3 cache generation. Not a reason to force all overlaps valid.
- Occurrence matcher implementation observed missing-module failure first, then passed tests for distinct occurrences, maximal public-name containment, equal/crossing ambiguity and token truncation. Complete inventory with new matcher:6,171 TRAIN/1,130 DEV observations recovered and zero lost-any-reference observations. Exact region relationship adapter likewise observed missing-module failure then passed; total61 focused/legacy tests. Parent V3 remains unchanged. Remaining Task3 work is actual feature/label rebuilding and shared live integration, not merely input flags.
- Ruling for next cache/live adapter: newly available singleton references can also lack old cached surface statistics. Those new references must explicitly use missing height in BOTH reconstructed data and live inference, determined from the legacy public resolver+mention eligibility for the same instruction. Do not silently use fresh depth only at inference. Legacy-eligible singleton references retain their existing height path.
- Public rebuild and private new-region supervision observed missing-module failures before implementation; a subsequent private-field injection test exposed accepted unknown network fields, fixed with an explicit public allowlist. Masked tokens cannot authorize restored references. Total73 focused+legacy tests pass. Actual CPU FrozenBERT revision86b5e0934494bd15c9632b12f734a8a67f723594 smoke passed on5 distinct source files, including all available TRAIN/DEV source schemas and a real restored group with one spatially suitable candidate. Existing4 singleton reference feature columns match exactly, original parent hashes unchanged, no CUDA context initialized. This is a small real integration check, not complete cache reconstruction or evidence of navigation gain.
- Follow-up spatial-field injection tests exposed two additional missing input-boundary checks; added one shared public spatial allowlist to both rebuilding and supervision. All74 tests now pass. Re-run actual five-source CPU smoke on the final checked adapter before publishing this change. Parent/runtime V3 remains untouched.
- Final actual five-source CPU smoke re-run passed after the input allowlists. V3 remote history now records all25 epochs (20+5), every TRAIN33440/DEV5647 population complete, no eligible checkpoint. Trainer2050932 exited; actual verification process2116210 is live and uses GPU, so do not launch any other GPU work yet. Verification/curves/confirmation handoff must finish or reject authoritatively; no forced correction. Next task remains full feature-cache reconstruction and live adapter integration for the new version.
- Full overlay builder/context/loader now implemented. Eight initial storage/role tests passed after missing-module RED; population checks and directory-binding tests subsequently exposed/fixed missing exact-population and original-directory protections. Dataset rejects sealed roles before source construction. Safe tensor loading, code/data/BERT snapshot hashes, immutable parent checks, final complete-population marker and CPU-only GPU exclusion protect building/resuming/loading. Total84 focused/legacy tests pass. Full output is revision_v4/reference_cache, separate from all old artifacts. Actual full construction and live integration remain pending until launched/verified.
- Actual CPU-only full rebuild launched as graphnav-a2-reference-v4-build-20261029.service, MainPID1241035, invocation e35147c8250543a19bf34141bb50aaa5. Observed real progress beyond2,200/39,087 views; no final manifest yet. Keep all cache-signed modules immutable while it runs. New standalone reference_perception.py wraps one original perception extraction and the exact shared public rebuilding adapter; episode binding accesses only public map/name queries and leaves legacy references/controller untouched. Two wrapper CPU tests pass, bringing the focused suite to86. Future V4 checkpoint/policy provenance and actual CUDA parity still required before GPU training/navigation; the existing V3 policy has not been switched.
- New reference_v4_protocol/train implemented separately with20+5 schedule, unchanged numerical V3 engine helpers, strict own-feature proof chain, exact initialized-weight fingerprint and per-epoch proof refresh. RED tests rejected missing/partial cache, V3/partial/foreign GPU/runtime proofs and metadata-only batch config changes. Reused training/epoch/model/loss/fusion tests broadened verification to131 passing CPU tests. V4 preparation/policy/verification files still need implementation and actual GPU checks; no V4 training admission yet.
- Actual adversarial-review helper requested a082102..b07a17c and terminated with workspace routing discovery timed out (log revision_v4/adversarial_reference_repair_20261029.log). No tool approval. Supplementary fresh read-only reviewer found generic query-formatting alias loss (zero actual cases in allowed TRAIN/DEV description superset) and unsafe trust of resumed overlay content without prior content journal. Current resumed0 generation remains valid on observed evidence. New non-cache-signed V4 admission now rejects ANY resumed v1 cache and binds final complete status SHA+manifest SHA+39087 population into all proofs. Generic resume/formatting API fixes are not claimed and cache-signed code stays immutable. Full data generation active, last observed beyond23,400/39,087.
- Supplementary review finalized: independently108 related CPU tests pass; no demonstrated GT leak or height-parity issue invalidating current fresh build. Reviewer confirms fresh-only admission blocks the demonstrated resume issue from this training. Both generic API defects remain documented; no adversarial tool approval is claimed. Next work: implement V4 preparation, actual policy/provenance and independent full-artifact verifier, complete data generation, then serial CUDA/runtime checks and full training.
