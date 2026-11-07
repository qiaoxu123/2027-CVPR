# Navigation Failure Attribution Implementation Plan

> **For agentic workers:** Work through this plan in order; preserve all existing checkpoints and rollout artifacts.

**Goal:** Attribute all 2,470 expanded-binding Val-Seen rollout outcomes and quantify how often spatially deduplicated Top-K already contains the target.

**Architecture:** Reuse the frozen per-step rollout traces and public query manifests. Add an offline analyzer that joins post-rollout GT only for scoring, reconstructs candidate ranks and valid historical RGB-D coverage, and writes per-task CSVs, summaries, plots, and an HTML case browser.

**Tech Stack:** Python, NumPy, pandas/CSV, existing GraphNav data loaders, Matplotlib, static HTML.

**Spec:** User request dated 2026-11-07, “导航失败归因与 Top-K 候选可纠正性分析”.

## Global Constraints

- Include all 2,470 Val-Seen tasks in denominators.
- GT is post-hoc diagnostic only; never pass GT to query binding, candidate generation, ranking, observations, or action selection.
- Do not modify Geometry, Controller, Stop, or checkpoints.
- Do not treat local soft decoding as spatial-NMS Top-K.
- Separate geometry footprint support from visual instance recognizability.
- Do not use future frames to assess observation available before a decision.

## Review Focus

- Missing per-step logits/candidate coordinates: classify as unavailable rather than infer from final NE.
- Multiple Polygons under one name: report ambiguity; do not select by GT.
- Historic RGB support: count only frames up to the target decision time and intersect with the camera footprint/depth validity.
- Main cause: make primary classes mutually exclusive while retaining overlapping evidence tags.
- High-level relation/object category: state dataset-provided label availability and mark unsupported stratifications unknown.

---

### Task 1: Audit rollout artifact coverage

**Files:** inspect existing `GraphNav/runs/v2_top1_public_binding_s42_20261107/full_navigation/` artifacts and public query/annotation metadata.

- [x] Verify every arm has 2,470 paired rows and inspect exact fields in each step diagnostic.
- [x] Record which requested quantities can be reconstructed without rerunning the policy.
- [x] Keep missing fields explicitly unavailable; identify only necessary non-policy diagnostics.

### Task 2: Offline all-episode attribution

**Files:** create a focused report-generation script under `scripts/` and results under `reports/unified_pgvb/top1_public_binding/navigation_diagnostics_20261107/`.

- [x] Compute per-episode failure phases, mutually exclusive primary cause, overlapping evidence tags, and spatially deduplicated target rank from the recorded beliefs/candidates.
- [x] Reconstruct causal historical RGB-D footprint/valid-patch support, marking geometry support as a proxy rather than instance recognition.
- [x] Analyze the 53 corrected and 21 damaged paired tasks from prior public-name binding comparison.
- [x] Export full-row CSV/JSON, aggregate charts, and a browsable HTML case page.

### Task 3: Validate and publish

- [x] Test full split coverage, pairing, no GT in inference inputs, temporal causality, and NMS/soft-decoder separation.
- [x] Write the Chinese report with limitations and ranked next-step options.
- [x] Commit only this diagnostic plan, scripts, and reports; preserve unrelated worktree changes; push the current experiment branch.
