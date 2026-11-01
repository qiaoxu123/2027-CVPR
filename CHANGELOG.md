## [0.2.0] - 2026-11-01
### Features
- Optional visual relation messages between observed candidates and up to 16 observed anchors (`--relation`).
- Explicit frame/BERT cache options and checkpoint evaluation with real/zero/shuffled visual controls and relation deletion.
### Design Rationale
- Reuse frozen language-conditioned visual features and relative normalized map coordinates; only the residual head learns. Sparse anchors limit pair expansion to O(N*16).
- Restore training from the locally validated reconstructed cache rather than historical absent paths.
### Notes & Caveats
- This is a lightweight relation feature prototype, not an AerialVG reproduction or supervised target–auxiliary object matcher. No explicit relation labels are used.
- Short prefix runs are smoke diagnostics and cannot establish navigation improvements. Checkpoint loader continues to reject smoke models for deployment.
- Existing baseline is unchanged unless relation is enabled; no future observations or target labels enter forward.

## [0.3.0] - 2026-11-01
### Features
- Optional instruction/geometry-conditioned relation attention, spatially diverse anchors and named-map contour association.
- Observed visual hard-negative ranking loss and full-Dev checkpoint selection with a separate short-training budget.
- Evaluation strata for observed targets, ambiguous observed candidates and relation-word instructions.
### Design Rationale
- Compare attention versus ordinary visual residual under the same ranking objectives to distinguish architecture from loss effects.
- Use public BERT node-name representations and contour distances; no fabricated instance correspondences or relation labels.
### Notes & Caveats
- Relation-word and ambiguity strata are proxies, not manual relation annotations. Geometry association does not guarantee a visually matched landmark identity.
- Keep prior checkpoints loadable; attention is opt-in. Training smoke weights remain excluded from formal closed-loop deployment.

## [0.4.0] - 2026-11-01
### Features
- Downloaded and tested actual official AerialVG source/checkpoint; full demo inference and relation/hierarchy/language interventions recorded separately.
- Added optional `--official-relation` adapter reusing the official pretrained K² pair-token module, with learned navigation feature/text projections and zero-initialized gain.
### Design Rationale
- Follow actual released relation implementation rather than treating generic candidate messages as an AerialVG reproduction.
- Preserve 256-dimensional official interfaces and Top-15 observed selection; freeze upstream relation weights while learning feature alignment.
### Notes & Caveats
- Official training recipe is not fully released; public dataset requires access approval. Demo interventions are not benchmark accuracy.
- Four aligned spatial pyramid maps required by official hierarchical attention are unavailable in existing pooled ROI caches; HCA is not claimed implemented in navigation.
- Prospective audit excludes grouped map/instruction examples from new-head training/selection, while the original frozen backbone has seen Train. It is not unseen-map evaluation.

## [0.5.0] - 2026-11-01
### Features
- Actual official decoder-query extraction from current CityNav views, retaining all900 candidates, full token features, public footprints and observation validity.
- Shared-validity NMS/geometric/spatial proposal selection and post-selection label audit.
- Native object-query ranker with null alternative and supported, zero-initialized goal-belief fusion scaffold.
### Design Rationale
- Equalized probe exposes proposal truncation and ranking separately; use actual aligned detection queries rather than projected pooled ROI features.
- Include both initial observations and causal Stage1 prefixes in grouped cache collection for later online validation.
### Notes & Caveats
- Small-probe coverage and software tests do not establish navigation gain. Native ranker training and closed-loop evaluation remain required.
- Geometry is orthographic footprint association with a vertical-depth proxy, not perspective occlusion reconstruction.

## [0.6.0] - 2026-11-01
### Features
- Grouped native object-query training cache with initial and causal prefix observations; actual HCA/decoder features remain aligned to their native text representation.
- Query/null classification plus supported goal fusion, optional official relation fine-tuning, shared initialization and BF16 interface regression coverage.
- Causal online encoder and navigation adapter; GT-free matched proposal selection and exact cache/online input validation.
- Isolated paired navigation entry point with native checkpoint identity and frozen held-out gates; archive-source hashing fallback in a local evaluator copy.
### Design Rationale
- Train-only diagnosis showed20m visual density kernels suppress useful fusion; use8m kernels and independently optimized nonnegative fusion gain.
- Keep original trajectory/Stop controls fixed in the first navigation comparison to isolate visual evidence effects.
### Notes & Caveats
- Prepared Dev128 is an offline development subset, not full navigation confirmation. Relation head is not assumed better than the plain native-query head.
- Current-only branch has no native temporal fusion yet. Dynamic goal stability and the existing exact-arrival guard require rollout evidence.
- Original shared evaluator, frozen checkpoints and previous formal results remain intact; no Git commit is fabricated for archived source.

## [0.7.0] - 2026-11-01
### Features
- Explicit Train-only on-policy navigation collection with grouped Dev exclusion and public native-query evidence snapshots.
### Design Rationale
- Collect the deployed policy's own near-goal and failure states to address teacher-prefix distribution mismatch before changing arrival decisions. Reuse the controller and scoring path; attach target labels only after rollouts.
### Notes & Caveats
- Train collection requires a completed explicit Train selection manifest; validation and Test gates remain in force. Collection metrics are training diagnostics, not navigation generalization evidence. The optional evidence sink is inactive in ordinary evaluation.

### Native temporal evidence experiment
- Add an opt-in causal residual window on the fixed world grid, averaging only observed support and resetting at episode start. Window1 preserves the current-only protocol; held-out evaluation requires the history window to match the frozen candidate.
- Add Train-only temporal ranking diagnostics and an on-policy/teacher dataset merger. Offline ranking or reduced candidate jumps do not establish closed-loop navigation gains.

## [0.8.0] - 2026-11-01
### Features
- Explicit validated native Seen evaluation resume, retaining complete episode-prefix bytes and hashes and appending only missing episodes.
### Design Rationale
- Recover an interrupted2470-episode run without discarding2047 complete trajectories. Replay saved traces solely for post-rollout aggregation; fresh policy episodes retain frozen controls and weights. Detached runner preserves process independence from the interactive session.
### Notes & Caveats
- Resume is restricted to incomplete frozen native Seen runs; completed outputs, changed selection/weights/controls and incomplete JSON lines are rejected. Timing metadata distinguishes resumed-session wall time/GPU peak from per-decision timings across both sessions. Actual2+2 episode smoke matches uninterrupted poses exactly.

### Publication compatibility
- Publish native shared evaluation as `scripts.evaluate_native_shared_closedloop`; preserve the existing resolution/residual evaluation entry point. Native launchers use the isolated entry point. All18 experiment test files:44 pass,3 asset-dependent skips.
