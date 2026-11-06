# PR26 main-Belief angular consistency execution

Goal: same Original Unified V2 initialization, full21878 TRAIN seed42 five epochs, three arms predicate-only/angle0.1/angle0.3; frozen BERT/PGVB/controller; three full2470 Val-Seen rollouts, no Test or main merge.

- [ ] Audit and test blind public bindings, Predicate0/east1/west2/north3/south4, conservative direct relation scope (negation/compound/heading/intermediate rejection), no GT eligibility or online angles.
- [ ] Pure geometric/circular loss checks, no-support handling, CPU suite; real CUDA total/angular-only gradients into both Geometry/Relation and Predicate embedding.
- [ ] Before fitting audit allTRAIN eligibility and GT agreement under existing engine axes (8bin and half-plane separately); do not flip axes or filter by GT agreement. No independent CRS claim.
- [ ] Five epochs A/B/C, identical seed/source/init/config; reuse frozen BERT cache, record samplewise loss components, gradients, resume state and frozen modules. Concurrency/resource usage documented.
- [ ] Freeze final checkpoints; GPU runtime TRAIN smoke, TRAIN Birmingham paired location/direction diagnostics (resubstitution only); full2470 Val-Seen for each, all failures included.
- [ ] Compare A vs matching public-bound full-text retraining47.37% to isolate Predicate; retain old45.75% context separately. Compare B/C vsA for angular contribution; no Val checkpoint selection/tuning.
- [ ] Actual heatmaps/curves/failures/report and scripts, tests, adversarial-review invocation, staged commits/push PR26.

World+X=0deg, +Y=90deg are map-plane axes. Cardinal semantic match to them remains unverified; existing rule convention not altered. GT only after forward in TRAIN losses/offline scores. All public Polygon binding uses original text and full map catalog, no target ID lookup or GT nearest Anchor.
