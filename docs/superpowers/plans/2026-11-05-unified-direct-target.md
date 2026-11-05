# UnifiedPGVB direct-target implementation and experiment plan

Scope: execute the requested TRAIN audit, joint optimization and frozen-policy
Val-Seen comparison on the existing experiment branch. No changes to Stable Goal,
Trajectory, Stop, horizon, historical weights or main. Preserve the pending
instance-memory analysis separately; do not treat it as target-training evidence.

1. Audit `structured_navigation.py` and `unified_instruction.py`. Reuse cached
   text-only dependency parses, rebind names to the actual public catalog, retain
   complete target text and attributes. Only forward, one-hop relations with
   resolved scope/reference frame enter training. Unknown relative headings
   abstain; a UAV yaw does not establish the instruction reference frame.
2. Produce TRAIN inventory with explicit rejection reasons and public/label
   separation. Split by maps and public anchor identities before RGB extraction.
   GT never decides parser eligibility, anchor selection or observation pose.
3. Extend UnifiedPGVB with an explicit trained direct-target entry point:
   integrated Geometry + full BERT tokens + polygon-relative relation context;
   current observed SigLIP patches and PGVB features predict target residuals
   only in supported cells. Absent RGB leaves geometry unchanged. Rebuild
   candidate IDs, coordinates and context together. Named-goal path unchanged.
4. Unit tests: parser scope, unknown frames, labels forbidden in forward,
   observed support, finite gradients, candidate alignment, checkpoint readiness.
   GPU smoke: real inputs, nonzero gradients, parameter updates and resume.
5. Train all eligible TRAIN fit queries, seed 42; hold out TRAIN maps for model
   selection. Optimize integrated geometry, relation and visual heads jointly;
   pretrained encoders may supply cached features, explicitly documented as
   encoder-frozen rather than full-backbone training. Save best/last, optimizer,
   RNG, source/data hashes, complete fit coverage and measured throughput.
6. Freeze configuration before official Val-Seen: deterministic first 32 eligible
   tasks and then every eligible task, without filtering on predictions or GT.
   Geometry / Geometry+Relation / Geometry+Relation+RGB share controller/protocol.
   Save exceptions as failures in denominator and disclose parser abstentions.
7. Chinese report: audit, tests, actual training/latency/VRAM, localization,
   paired closed-loop results, heatmaps and real trajectories. Commit/push to
   experiment branch only. Failed or unsupported stages explicitly incomplete.

Acceptance: empirical semantic displacement and RGB incremental benefit; do not
equate public Polygon registration or 20m proximity with verified pixel identity.
