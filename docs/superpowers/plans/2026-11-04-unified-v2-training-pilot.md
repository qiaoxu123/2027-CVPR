# UnifiedPGVB-v2 real training Pilot

Base main ce2c6dae. Preserve original branches/checkpoints.

- Environment/SHA -> V2 migration -> 29 existing tests -> two real GPU landmark smoke rollouts.
- Audit all TRAIN instruction bindings and complete reusable visual cache. Report original instruction universe separately from available supervised frame/query pool. Check TRAIN-only source and city/map/name/RGB isolation.
- Dedicated V2 trainer calls UnifiedPGVB.forward and saves full V2 state. Train pgvb only; freeze geometry, BERT/SigLIP2, controller, joint/direction/target. Cached heads need no redundant static Geometry forward at every update; independently measure it in runtime.
- seed42 BF16, batch frame groups, workers configurable, clipping, periodic last/best, exact sampler/RNG recovery. Synchronized forward/backward/optimizer/data/wall timing, GPU telemetry, finite gradients and optimizer-state restoration.
- A200 then resume B+1000; if cheap, finish current epoch and one full additional cached TRAIN epoch, recording actual work. Never describe cached pool as all 22,002 original observations. No auto10 epochs.
- Pre/post independent TRAIN city evaluation and fixed Geometry-only control; public Polygon labels are weak geometry, not human segmentation. Fresh SigLIP extraction benchmark and deterministic cache compatibility test, Building/Road/multiPolygon plots.
- Two post-training runtime smokes (ValSeen diagnostic tasks never used to fit/select models). Save configurations, curves, metrics, costs, checkpoint paths/SHA, Chinese report and commit/push.
