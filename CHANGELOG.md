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
