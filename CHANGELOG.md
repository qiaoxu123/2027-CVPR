## [Privileged complete-landmark RGB experiment] - 2026-11-05

### Features
- Add isolated all-TRAIN public-Polygon crop inventory, frozen SigLIP2 feature cache and multiscale PGVB visual training/evaluation.
- Separate privileged checkpoints, held-out city predictions, RGB controls and target labels from ordinary navigation.

### Design Rationale
- Distinguish coverage and sampling density from visual evidence; average repeated orthophoto features on the existing world grid.

### Notes & Caveats
- Source orthophoto crops stay on the server pending data-license confirmation. Position metrics are not instance recognition accuracy.
- Frozen Geometry/Relation already saw TRAIN; held-out maps isolate only new visual fitting. No controller or Stop changes.

## [Full-relation TRAIN execution and evaluation] - 2026-11-05

### Features
- Add deterministic within-epoch recovery, periodic optimizer/RNG checkpoints and full TRAIN coverage logs for PR17.
- Record complete Val-Seen step Beliefs, actual RGB SHA, candidate/Goal and Stop diagnostics; retain failures and no-reference tasks.
- Add an isolated original-Geometry baseline and recover interrupted JSONL writes without dropping complete episodes.

### Design Rationale
- Separate changes in jointly trained Geometry, learned relation and observed PGVB evidence using matched full episodes.
- Eliminate globally padded reference columns without changing the masked attention model.

### Notes & Caveats
- BERT/SigLIP2 stay frozen per PR17; PGVB uses the available biased TRAIN RGB subset, not all-instruction visual training.
- No Stop/Goal threshold changes, no Val/Test training, no new explicit geometric rules.

## [Direct-target evidence safety] - 2026-11-05

### Features
- Rebind direct-target anchors to the current public Catalog, handling integer
  runtime IDs and string JSON IDs without destination-dependent selection.
- Separate effective Geometry fallback fields from rejected directional proposals
  in rollout logs and Belief visualizations.
- Add TRAIN synchronized direction probes, identity ambiguity audits and
  action-equivalent Stable Goal checks. Stop/Goal thresholds unchanged.

### Design Rationale
- Anchor RGB discrimination is provisional evidence, not target-instance identity.
- Public names/map objects are binding authority; retain homonymous union geometry
  while abstaining from ambiguous visual identity. Keep all evaluation tasks.

### Notes & Caveats
- Direction head remains unverified; no geographic axis flip from target labels.
- Preserve original Checkpoint/reports. Fixed56 is not full CityNav SR.

## [UnifiedPGVB direct-target pilot] - 2026-11-05

### Features
- Add strict public-text/catalog direct-relation eligibility with explicit scope
  rejection, original BERT token conditioning and separate target-label files.
- Add an opt-in integrated Geometry/PGVB direct-target head, real RGB cache,
  joint core training/resume and original-target paired closed-loop diagnostics.

### Design Rationale
- Keep trained landmark navigation and Stable Goal/Trajectory/Stop unchanged.
- Local target residuals affect only supported observed world cells; absent
  observations retain the relation prior. Rebuild candidate IDs/XY/context together.
- Use map-isolated TRAIN model selection; freeze Val-Seen eligibility/configuration
  before paired diagnostics. The encoders are cached/frozen; core networks train.

### Notes & Caveats
- Automatic dependency relations and projected Polygons are not independent pixel
  identity truth. Relative reference frames abstain when unknown; cardinal world
  convention retains the earlier unverified geographic interpretation.
- The initial strict subset covers a small fraction of CityNav instructions.
  Partial-task SR is never a full-dataset SR claim. No Test or new Stop thresholds.
- The direct-target head requires a separately trained checkpoint. Merely enabling
  a flag cannot turn the historical untrained target head into a validated model.
