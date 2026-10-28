# A2 V4 Independent Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the complete V4 20+5 run to independently calibrated, frozen A0/A2 evaluation and verify whether A2 exceeds A0.

**Architecture:** Add separate V4 post-training modules; reuse pure V3 population, packet and statistical operations where their contract is unchanged. Preserve all training-bound files and root tests, all original results, and all confirmation target assignments. No global monkeypatching or relabelling V3 predictions as V4.

**Tech Stack:** Python, PyTorch, existing native CityNav runtime, pytest, systemd serial runner.

**Spec:** `docs/A2_WALDO_LANDMARK_RELATION_BELIEF_20261027.md`; user full-evaluation and no-Unseen-tuning instructions; `2026-10-29-a2-road-reference-repair.md`.

## Global Constraints

- Seed 0; frozen A0, controller, WALDO, BERT and SigLIP2; 384 RGB and 28×28 field.
- Full20 ranking +5 refinement, independently verified before any independent confirmation observations.
- Same predeclared 163 fresh targets:84 FIT,79 VERIFY;130 old sealed VERIFY;209 VERIFY total.
- Consume no VERIFY performance before one FIT choice is frozen; never select a runner-up using VERIFY.
- Preserve source/role/feature/checkpoint/code byte-hash lineage; fresh calibration tensors must use actual V4 public-reference extraction.
- No edits to training CODE_FILES or root `tests/`; new tests in `tests_confirmation_v4/` and `tests_navigation_v4/`.
- One GPU task at a time; no benchmark tuning; full Seen2470 and Unseen2697 only after parameter freeze.

## Review Focus

- Reusing old confirmation metadata must preserve target assignments without consuming old tensors or changing V3 files.
- Changed training feature cache must invalidate calibration, including disabled/fallback paths.
- Failure or missing training proof must prohibit collection, not silently fall back to epoch0 or V3.
- VERIFY refusal must leave heldout data unopened and prohibit a second threshold choice.
- Full comparison must preserve episode IDs, controller and A0 bytes; partial runs cannot become final reports.

### Task1: Independent metadata and admission

Files: create `reference_v4_confirmation_protocol.py`, tests `tests_confirmation_v4/test_protocol.py`.
Interfaces: `protocol_metadata(roles, source, seal, *, binding, source_sha256, seal_sha256, expected=COUNTS) -> dict`; `prepare() -> dict` writes immutable V4 metadata only. Reuse pure `build_population` and opaque byte-hash checks.
- [x] Write tests rejecting supervised overlap/query reselection/incomplete feature binding; assert fixed fresh/sealed queries survive and protocol is feature-bound.
- [x] Run new tests and observe missing implementation failure.
- [x] Implement pure metadata and live prepare; preserve all V3 files and source seals.
- [x] Run CPU tests; do not collect any observations.

### Task2: V4-only independent observation capture

Files: `reference_v4_confirmation.py`, `tests_confirmation_v4/test_capture.py`.
Interfaces: `collect()` admits only verified eligible V4 best.pt;163 fixed fresh A0 shadowOFF trajectories; preserves RGB and public-reference geometry. Reuse unchanged capture/commit validation pure functions, actual `ReferenceV4Policy`.
- [x] Write and fail admission/cross-version/fallback tests.
- [x] Implement version-bound collection with exact A0 command parity, destination access only after each complete rollout.
- [ ] CPU tests and separate real TRAIN functional proof before heldout collection.

### Task3: FIT packets, sealed replay and one VERIFY

Files: separate `reference_v4_calibration_data.py`, `reference_v4_calibrate.py`, `reference_v4_sealed_replay.py`; tests in `tests_confirmation_v4/`.
Interfaces: feature-bound packet datasets, immutable FIT choice and analysis signatures compatible with `reference_v4_provenance.validate_gate`.
- [ ] Test role barriers, population completeness, stale feature/packet rejection and no-candidate VERIFY refusal.
- [ ] Implement public-only rebuilt old observations; recompute real V4 predictions, never reuse V3 predictions; sealed labels only after frozen FIT choice.
- [ ] CPU suite; serial actual collection/FIT/replay/VERIFY only after eligible training proof.

### Task4: Frozen full navigation and reports

Files: separate `reference_v4_navigation*.py`, tests `tests_navigation_v4/`; reports and curves.
Interfaces: immutable validated calibration + full checkpoint;64 TRAIN pilot; full Seen2470 then Unseen2697 A0/A2, resumable episode commitments.
- [ ] Fail tests for partial population, stale choice, cross-version model and mismatched A0/controller.
- [ ] Implement serial evaluation preserving baseline; record all requested metrics, map breakdown, paired differences, candidate coverage, latency, fix/harm/fallback.
- [ ] Verify all episodes and artifacts; commit/push code and report summaries; retain large weights server-side.

## Execution ledger

- Authorized continuation uses inline implementation; no approval pause is needed for reversible local code/tests or already requested branch push.
- Active remote scientific commit stays ac03d24; subsequent metadata/code additions remain outside its frozen training inventory.

- Task1: observed eight missing-module test failures, then10 new V4 tests passed including full293-target metadata fixtures, persistence and no tensor-decoding checks. Supplementary independent review found no concrete defects; its missing integration coverage observation was addressed by the final two tests. Actual adversarial companion review is separately running.

- Task2: observed12 missing-module test failures; then13 collector tests passed. Full CPU regression612 passed,1 pre-existing rasterio warning in8.26s. Actual collector is not yet executed on independent data; collection remains prohibited until eligible complete25-round proof. Supplementary reviewer found no concrete defects and23 targeted tests passed; whole rollout/resume still awaits actual admitted execution.
- Actual adversarial metadata review finished exit1 after repeated network reconnections: workspace routing discovery timed out, no valid structured verdict. No tool approval is claimed and identical route is not retried.

- Task3 first component: new `reference_v4_calibration_data.py` implements actual-V4 fresh-record loading, FIT packets and frozen-choice barrier. Six initial tests failed on missing implementation; final nine data tests pass, including full synthetic163-target constructors,84 FIT packets and no VERIFY decoding. Supplementary review found a missing trained-checkpoint comparison in the FIT cache; coherent-commit regression failed before fix and passes after comparing packet signature -> analysis -> actual best.pt bytes. Full regression621 passed,1 pre-existing warning in15.81s. No new live independent data consumed.
- Frozen training inventory check: all48 CODE_FILES match ac03d24 bytes locally. New post-training code and tests remain outside that inventory. Sealed V4 public-reference rebuilding, final calibration runner and serial post-training scheduler remain incomplete and must be implemented before actual confirmation.

- Task3 second component: added `reference_v4_calibrate.py` and `reference_v4_confirmation_preparation.py`, binding analysis/results to V4 model/features and actual post-stage CPU logs. Three calibration tests and four CPU preparation tests first failed on missing modules. Supplementary review found misleading pilot-pending status on failed VERIFY; regression failed before fix, then rejected and passed cases gained distinct statuses. Added direct refusal-to-regenerate tests after capture/FIT. Final full CPU regression632 passed,1 pre-existing warning in17.90s. Sealed V4 replay and serial scheduler remain pending; no independent data consumed or final-gate enablement.
- Remote full epoch1 completed33440 TRAIN +5647 DEV, checkpoint77b06e17732bd1dfc66937745e179347f6ba2331bc584b2ecfe38b0a95e6072b. Same1029 real DEV observations: V4 proposal recall.216715 vs V3 epoch1 .136054, but fixed ungated map recall.500486 below A0 .653061, corrected39/broken196. Internal diagnostic only; ranking epoch1 cannot prove navigation gain or justify early ending of full25.
