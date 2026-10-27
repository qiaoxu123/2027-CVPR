# Local Semantic Relation Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement independently trained, language-conditioned box/reference local refinement with original-paper runtime integration.
**Architecture:** Frozen BERT/SigLIP/WALDO/main policy; train depth CNN and two 256-dimensional language-conditioned relation blocks. Private labels remain separate, bounded local mass-preserving feedback precedes one original controller call.
**Tech Stack:** Existing graphnav conda Python/PyTorch, numpy, OpenCV, frozen HuggingFace models, existing WALDO vendor.
**Spec:** /home/20261024_1/Workspace/2027-CVPR/Experiments/BoxRelationCorrection/docs/superpowers/specs/2026-10-27-local-semantic-geometry-refinement-design.md

## Global Constraints
- Keep original 28×28 checkpoint, sources, flight controller, cached experiments unchanged.
- Group splits by target instance; no private GT in inference inputs; height proxy is weak surface evidence.
- Unknown identity is not a ranking negative; absent depth is not physical absence.
- Empty/no reliable/zero feedback returns original field; outside local support probability unchanged.
- No claim of subgrid navigation precision or improved metrics from implementation tests.

## Review Focus
- Empty candidate/reference lists and masked language must never produce NaNs (Task 2).
- Same-class dense boxes / uncertain annotation matches must not become false negatives (Task 1).
- No supported coarse grid center / all depth missing must allow correct fallback (Tasks 1/3).
- Frozen cache/checkpoint/code mismatch must fail before inference/training (Tasks 3/4).
- Partial preparation or insufficient calibration must not enable an untrained repair policy (Task 4).

## Task 1: Geometry, labeling and local fusion
- [x] Write failing tests for coordinate/height validity, candidate/reference permutation, uncertain-label loss masking, mass conservation and exact zero fallback.
- [x] Implement schema.py, geometry.py, labels.py and fusion.py. Validate finite values, physical units and masks.
- [x] Run CPU tests and record results.

## Task 2: Relation network and objective
- [x] Write failing tests for N/L=0, semantic and distance influence, depth-mask invariance and finite training gradients.
- [x] Implement model.py with frozen-feature inputs, two relation blocks, triad features, expected relation/rank/offset/matchability heads and masked objective.
- [x] Run CPU forward/backward and invariant tests.

## Task 3: Real observed extraction and original policy routing
- [x] Write failing integration/selection tests; implement common.py, perception.py, prepare.py and policy.py using new schema and explicit public-only inputs.
- [x] Extract real samples; inspect automated label coverage and run backward on real rows. Manual association validation remains a research prerequisite, not claimed complete.
- [x] Verify original checkpoint/source hashes and zero-branch original controller parity.

## Task 4: Training, calibration, evaluation, documentation
- [x] Implement train.py, calibrate.py, evaluate.py, visualize.py and verify.py; test checkpoint schema/splits and no-positive masks.
- [x] Run actual two-stage short training, calibration/fallback and paired runtime smoke; report coverage and limitations without efficacy claims.
- [x] Run independent whole-branch review plus actual adversarial-review; fix material findings with regression tests. Plugin timed out; independent review found3 fixed material issues.
- [x] Complete README/CHANGELOG and source ledger. Commit/push and remote bundle verification are the user-added final delivery step.

User explicitly authorized execution after reviewing the design. Execute inline without another approval round.
