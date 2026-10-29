# A2 Target-Balanced Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a separately versioned A2 V5 model with balanced contribution from existing target groups and determine whether it improves real-DEV candidate localization.

**Architecture:** Keep V4 data, model, loss, optimizer, and schedule intact. Add a wrapper that adjusts only per-row loss weights using source-family/map/target counts, then run a separately signed V5 trainer and verifier in a new output directory.

**Tech Stack:** Python, PyTorch, pytest, existing V4 experiment utilities.

**Spec:** `docs/superpowers/specs/2026-10-29-a2-target-balanced-training-design.md`

## Global Constraints

- Use only existing TRAIN/internal DEV data and never load benchmark inputs.
- Preserve the 8:1 real-to-teacher aggregate group weight and V4 mean row-weight scale.
- Freeze A0 and all existing encoder/controller assets; preserve the 20+5 schedule.
- Keep the existing eligibility gate unchanged; do not force enablement.
- Keep GPU tasks serial and never overwrite V4 results.

## Review Focus

- A repeated target must not gain weight merely because it has more observations.
- V1 and V2 source rows for the same map/target must stay distinct source groups.
- Invalid or non-TRAIN rows must fail before any cached tensor is read.
- DEV evaluation must not receive training weights.
- Weight normalization must preserve the previous average loss scale.

---

### Task 1: Target-balanced weight calculation

**Files:**
- Create: `experiments/a2_waldo_landmark_belief/reference_v5_weighting.py`
- Test: `tests/test_a2_reference_v5_weighting.py`

**Interfaces:**
- Produces `target_balanced_weights(entries, real_weight=8.0, teacher_weight=1.0) -> list[float]`.
- Each `entry` is a TRAIN manifest row with `source_kind`, `map_name`, `object_id`, and `role`.

- [x] Write tests for equal aggregate weight per group, 8:1 group mass, preserved mean weight, and rejection of non-TRAIN/unknown/malformed entries.
- [x] Run the focused tests and confirm the missing-module failure.
- [x] Implement the pure weighting function and make the focused tests pass.

### Task 2: Dataset wrapper and source audit

**Files:**
- Create: `experiments/a2_waldo_landmark_belief/reference_v5_data.py`
- Test: `tests/test_a2_reference_v5_data.py`

**Interfaces:**
- Produces `TargetBalancedReferenceViews(role, directory=...)`, a dataset wrapper over `ReferenceOverlayViews`.
- TRAIN items replace only `labels.sample_weight`; DEV items preserve their original unit weight.

- [x] Write tests showing TRAIN weights are attached by row order and DEV weights remain unchanged.
- [x] Run focused tests and confirm failure.
- [x] Implement the wrapper with source and manifest checks; pass tests.

### Task 3: V5 trainer and independent artifact verifier

**Files:**
- Create: `experiments/a2_waldo_landmark_belief/reference_v5_protocol.py`
- Create: `experiments/a2_waldo_landmark_belief/reference_v5_train.py`
- Create: `experiments/a2_waldo_landmark_belief/reference_v5_verify.py`
- Test: `tests/test_a2_reference_v5_train.py`
- Test: `tests/test_a2_reference_v5_protocol.py`

**Interfaces:**
- V5 artifacts live under `revision_v5/training/`.
- V5 signature binds V4's frozen input proofs plus V5 source hashes, weight formula, and exact TRAIN group counts.
- The verifier replays all 25 checkpoints against the unweighted DEV population.

- [x] Write tests that reject changed weights and incomplete/foreign 20+5 verification proofs.
- [x] Run focused tests and confirm failure for missing code and invalid weight/proof cases.
- [x] Implement a separate V5 trainer/verifier without modifying V4 source or artifacts.
- [ ] Run source-bound CPU admission and an actual weighted CUDA smoke test after the current serial V4 queue finishes.

### Task 4: Run and assess V5

**Files:**
- Create: V5 source-bound run records and report under `runs/a2_waldo_landmark_20261027/revision_v5/`.
- Modify: `CHANGELOG.md`.

- [ ] Wait until the current V4 verification, confirmation, navigation admission, and queued CPU diagnostic are terminal.
- [ ] Start V5 training serially on the GPU and complete all 20 ranking plus 5 refinement epochs.
- [ ] Independently verify all checkpoint outputs and select only by existing internal DEV rules.
- [ ] If no checkpoint is eligible or real-DEV proposal quality does not improve, stop this iteration and record the diagnosis; do not calibrate or benchmark an ineligible model.
- [ ] If eligible, run independent calibration, freeze all parameters, then execute the full paired Seen/Unseen navigation benchmark with the predeclared protocol.
