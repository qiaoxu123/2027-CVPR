# A2 Revision Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task with tests and independent review.

**Goal:** Build and validate a revised A2 that can outperform frozen A0 on complete Seen/Unseen navigation.
**Architecture:** target-disjoint verified full-category supervision, spatial proposal training, current-A0 context and bounded global-mass fusion. Original artifacts untouched.
**Tech Stack:** existing graphnav conda/PyTorch stack, cached tensors, CUDA single GPU.
**Spec:** docs/superpowers/specs/2026-10-28-a2-revision-design.md

## Global Constraints
Seed0; frozen A0/WALDO/BERT/SigLIP2/controller; grid28; original datasets/results immutable; full20+5; no unknown identity negatives; no benchmark tuning; serial GPU/flock; OFF on inadequate evidence.

## Review Focus
- Unknown/cross-target negatives must not leak labels across target partitions.
- Empty candidates/references/support must return exact original logits without NaNs.
- Spatial loss may use GT but forward/runtime inputs cannot contain target fields.
- Revision loader must preserve extraction provenance and original bytes.
- Existing opened calibration cannot be reused as untouched confirmation; no repeated Unseen selection.

### Task1: Context and bounded fusion
- [x] Add failing tests in tests/test_a2_revision_fusion.py for outside-peak correction, exact fallback, normalization/support and gradients.
- [x] Implement revision_fusion.py: a0_candidate_context(spatial), proposal_distribution(...), bounded_belief_mix(base,proposal,gate).
- [x] Run tests; CPU TRAIN oracle verifies changed mechanism actually can correct cases old mass conservation could not.

### Task2: Data and new network
- [x] Reserve256 unused original dev/calibration targets before new labels/training; exclude old cached target identities.
- [x] Tests for revision_data labels, cross-category supervision and original/derived target isolation.
- [x] Implement revision_data.py derived loader; revision_model.py new instruction category, A0-aware rank/trust/benefit heads, zero offset.
- [x] Add synthetic gradient/no-GT tests and all-row data audit. Do not modify old model/labels/fusion files.

### Task3: Loss, trainer and readiness
- [x] Implement revision_loss.py and revision_train.py explicit full20+5, proposal/map/benefit/risk supervision and full-pool selection; resume/hash/artifact guards.
- [x] CPU suite, actual CUDA TRAIN gradient gate, independent adversarial review; pin code/data.
- [ ] RUNNING: Train under shared flock; supervise actual unit/process and epoch/checkpoint/TB. Export curves and commit/push summaries.

### Task4: Independent calibration and real pilot
- [ ] Preserve new unused TRAIN confirmation targets and split fit/verify before sampling; fixed candidate first.
- [ ] Confirm updated runtime current-A0 context/fusion matches offline code. Test causal repeated observations and exact fallbacks.
- [ ] Independently calibrate risk, freeze, paired real A0/A2 internal pilot. Failed evidence implies OFF/no launch; revise only using TRAIN development data.

### Task5: Complete confirmation and publication
- [ ] After internal pass and immutable freeze, complete A0/A2 Seen2470 and Unseen2697, same episodes/controller; resumable raw traces.
- [ ] Full metrics, maps, paired statistics, coverage/latency/fallback, curves and A0/A1 preserved conclusion.
- [ ] Independently rescore, verify hashes/remote commits, audit success. Mark goal complete only if actual objective proven.

## Execution ledger 2026-10-28
- Implementation fa44c8a: full data/340CPU/CUDA/review passed; full20+5 started in graphnav-a2-revision-v1-train-20261028.service with immutable code/data pin.
- Runtime/artifact verifier 5af035d: 344CPU pass, actual cache parity0 error; actual frozen-model CUDA runtime and controller/pilot still pending.
- Current first6 completed epochs: latest dev cached-map R1 .69965 vs A0 .68644, corrected89/broken28; this is NOT navigation SR or final confirmation.
- Never edit pinned training files while unit live. New confirmation/calibration/pilot/full-evaluation files remain to implement.
