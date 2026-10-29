# A2-V6: frozen-evidence relational candidate diagnostic

## Scope
V5 proposal DEV R@1 was 33.63% vs frozen A0 68.04%; internal results alone do not establish navigation gains. This version introduces an isolated eight-neighbor spatial graph reranker over existing WALDO/SigLIP2 candidate features, candidate-to-landmark geometry and A0 prior evidence. No changes to A0, A1 or V5; no runtime integration.

## Safety and provenance
- Candidate residual score head is zero-initialized; original score is exactly preserved at init.
- Frozen V3 evidence encoder, all features label-free during forward.
- Weak spatial utility is a 20m support label, **not verified physical target identity**. Unknown candidates stay unknown in supervision.
- No-target head estimates whether a *spatially useful* candidate exists, not whether a visible object exists.
- Belief correction is default-disabled, maximum transferred mass is 0.10, with exact A0 fallback when disabled or rejected; prototype thresholds are NOT statistically calibrated.
- A target-ground-truth oracle is for TRAIN/DEV diagnostics only. It must never be passed into runtime or used on benchmark data.
- V5 checksum guards, role exclusions and frozen asset verification must be retained before GPU testing.
- Do not run full CityNav Val-Seen, Val-Unseen or Test-Unseen until independent calibration confirms safe gain.

## Local instructions
Run CPU tests:
    python -m pytest -q experiments/a2_waldo_landmark_belief/v6_tests

Then integrate a bounded 2-epoch TRAIN/DEV diagnostic using TargetBalancedReferenceViews('train'), ReferenceOverlayViews('dev'), initial_model() and collate_views. Train only RelationalGrounder non-frozen parameters. Report initial-vs-trained proposal R@1/NE for 5,647 DEV views and 1,029 real subset; compare with V5 and frozen A0. Report oracle candidate coverage, A0-wrong recoverability, and alpha=.10 oracle-correction ceiling separately. Save proof hashes and separate TRAIN/DEV identity/role protocol. Do not select based on sealed calibration or benchmark results.

## Gate to further work
If DEV proposal does not beat V5 33.63%, halt, inspect hard negatives and target/landmark identity. If it improves, add verified TRAIN-only identity correspondences and a learned risk gate; use independent groupwise confirmation before admitting any closed-loop runtime. No claim of SR gain from offline DEV.
