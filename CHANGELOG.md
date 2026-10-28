# Changelog

## [0.6.0-a2-remote-execution-dev] - 2026-10-28
### Features
- Prepare the user-requested RTX5090 migration with exact-file dataset/model reuse, direct remote code/public-weight retrieval and an isolated environment; stop and preserve the local untrained waiters before migration.
- Keep GPU admission exclusive by default. An explicit source-bound policy may exempt only user-approved existing background services matching UID, executable argument, complete command digest and instance cap; any other A2 or unknown process still blocks. Never stop other users' services.
- Use the actual user UID for the calibration scheduler's systemd runtime directory rather than a machine-specific UID.
- Add a locked resumable host bootstrap, explicit execution-target requirement and prebound reviewed-source/input manifests. Preserve orphan formal training records and require current CPU/CUDA/model checks before training.
- Use a trusted absolute GPU-query binary and host-bound optional service exemptions.
### Design Rationale
- The new host already serves speech models. Resource admission must distinguish those explicitly approved services from another experiment, while retaining serial A2 execution and unchanged data/model/controller/20+5 configuration.
### Notes & Caveats
- Old untrained V3 source proofs and initial tensors were archived before changes. They are historical evidence, not current remote authorization: rebuild CPU/CUDA/functional proofs on the new host and require the same initial-weight fingerprint before training.
- Migration and transfers remain in progress. No remote formal epoch or final A0/A2 improvement is claimed. Background speech traffic and different hardware must be disclosed with latency results.

## [0.5.0-a2-onpolicy-relation-dev] - 2026-10-28
### Features
- Preserve v2's failed FIT calibration, with no VERIFY performance aggregate, and version the next target-disjoint data roles: TRAIN33440, DEV5647, independent FIT84 and VERIFY209 targets.
- Add untrained evidence-only relation/ranking and risk-head prototypes; candidate ranking excludes A0 prior shortcuts, offsets stay frozen, and observed4m kernels use stable direct coordinate differences.
- Add source-bound full TRAIN/dev loader and spatial/category utility labels, with exact canonical observation populations and role-SHA guards. Unknown object identities remain unlabelled.
- Add full-pool utility ranking/trust, relation/category/proposal losses and fixed-action refinement losses; risk-only gradients cannot change the evidence actor, and real-state rows receive absolute8x weight.
- Integrate the full20+5 trainer, stage21 best-rank restore/fresh optimizer, byte-bound resume/aliases, real/teacher DEV metrics, actual per-epoch inference packets, TensorBoard, all25 DEV/optimizer/artifact verification and exported curves.
- Add isolated CPU and real-TRAIN CUDA gates,32-target A0/OFF/ON functional preflight with full native belief checksums, and a serial persistent GPU-idle scheduler. Untrained functional policies are limited to signed TRAIN queries; formal enabled gates require independent84FIT/209VERIFY evidence.
- Prepare the fixed163fresh confirmation inventory (84FIT/79VERIFY) plus the opaque existing130VERIFY seal. Add a full-training-gated shadowOFF collector with actual RGB/source hashes, post-whole-rollout labels, partial replay archives and actual-inference packet checks.
- Add the serial post-training service: finish all25 verified epochs first, collect the fixed163 queries, select exactly one rule on84FIT, then replay the sealed130 only for a nonempty choice and evaluate that same rule on209VERIFY once. Preserve rejected decisions without opening heldout performance.
- Bind the post-training implementation and its separate27 tests to a456-test CPU proof. Validate committed FIT packets, restore an interrupted decision seal, and retain authoritative successful predecessor service state without restarting training.
- Fix the64-query real-navigation pilot inventory before training, preserve original reserved209 VERIFY source queries, and add root-bound admission that rejects missing/mutated candidate commits before validating complete training and the unique calibration lineage. Navigation-stage tests remain outside both queued source/test inventories;468 combined CPU tests pass.
- Implement actual serial A0/V3 navigation for the fixed64 pilot and full2470/2697 populations, raw native RGB/784-cell beliefs, source/controller/metric rescore, inference-only synchronized latency and verified complete-output retention. Freeze the unique calibrated rule only after the full safe changed pilot; bind all consumed artifacts and actual local frozen encoder bytes. Add public-geometry crosschecks and a synthetic64 end-to-end CPU integration;485 combined CPU tests pass.
### Design Rationale
- Opened FIT diagnostics found more tested single-candidate localization opportunities than current mixed proposals. Real frozen-A0 states supplement teacher-state training; unknown object identities remain unknown.
### Notes & Caveats
- V3 remains untrained. All39087 TRAIN/dev rows passed the integrity-hardened loader audit (386.68s);429CPU tests and six real-TRAIN CPU backward updates passed. Trainer/runtime are implemented; actual CUDA smoke/preflight, full20+5, independent confirmation and full navigation remain incomplete. No V3 performance or final A0/A2 full-benchmark result exists.
- Existing130 VERIFY targets were observed and machine-preprocessed, but not used for comparative performance feedback. The next model must exclude all calibration supervision and statistics, then freeze before one heldout aggregate.
- Post-training-stage tests live separately in tests_confirmation_v3 so the queued training's429-test source proof stays immutable. Combined CPU validation is456passed; both training and confirmation services are live and waiting. No new confirmation observations or heldout performance feedback have been consumed. Actual GPU/CPU packet tolerance, the64-query real navigation pilot and final navigation still require real execution.

## [0.4.0-a2-direct-risk-dev] - 2026-10-28
### Features
- Preserve completed v1 relation20+5 and its rejected independent confirmation; add isolated public prior/proposal risk features and direct neutral/gain/harm head primitives.
- Reserve256 new never-observed TRAIN-source targets; add complete frozen originalTRAIN/dev probe extraction, integrity checks and opened-FIT transfer diagnostics.
### Design Rationale
- v1 safe gate improved only3heldout groups with confidence interval starting0. Direct fixed-action outcome supervision and actual safe operatingpoint selection address the demonstrated mismatch between AUC/distance scores and deployed correction.
- Keep completed relation/proposal behavior frozen; additional head20+5, new independent confirmation and full navigation remain required. Balanced scores are not probability guarantees.
### Notes & Caveats
- Risk head completed actual real-TRAIN CUDA backward and all additional20+5epochs;1550updates and all25dev-checkpoint recomputations verified. Independent256-target/2055-view confirmation completed; all30FIT gates failed, enabled=false and no VERIFY performance aggregate was invoked. Final navigation remains incomplete. Diagnostic classifier AUC does not prove navigation improvement.
- v1 original code/weights/calibration and A0/A1 remain unchanged. No new benchmark evaluation or forced gate has occurred.

## [0.3.0-a2-revision-dev] - 2026-10-28
### Features
- Add isolated fresh-seed A2 revision with trusted cross-category negatives, full-pool spatial proposal supervision, current-A0 context, instruction category consistency and a separate benefit gate.
- Replace candidate-support mass conservation with a bounded global mixture (maximum gate .25), exact OFF fallback and zero offsets.
- Add full20+5 revision trainer, atomic checkpoints/resume, proposal/map-based selection, pinned CPU/CUDA audit gates and 256 reserved independent confirmation targets.
### Design Rationale
- Existing A2's known-only ranking did not reflect full-pool competition; support mass conservation prevented most recoverable errors from changing A0's peak.
- Grid position supervision uses TRAIN coordinates only in loss, while unknown candidate identities remain unlabelled. All foundation models/controller and28x28 grid remain frozen.
### Notes & Caveats
- Existing A0/A1/A2 weights and reports are unchanged. Oracle correction capacity and smoke gradients are not trained navigation gains.
- Original calibration has been inspected; independent confirmation uses separately reserved never-observed targets. Val-Unseen remains final confirmation only.

## [0.2.0-a2-dev] - 2026-10-28
### Features
- Honor user-requested collection stop with an immutable completed-episode snapshot, secondary target-disjoint train/dev/calibration split, load-time cross-partition label masking and unchanged full20+5 epoch schedule.
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
