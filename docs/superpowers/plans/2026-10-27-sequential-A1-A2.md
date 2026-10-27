# Sequential A1 Validation → A2 WALDO–Landmark Belief

User instruction: finish A1 fully before starting A2. Strictly one GPU experiment at a time; routine supervision every five minutes. Preserve original A0/A1 results. A0 is the existing complete formal archive, not the cancelled partial retraining.

## Phase 1 — A1 gate (current branch research/joint-rgb-geometry-retrain-20261027)
- [x] A0 retraining cancelled and preserved separately; archived A0 checkpoint SHA verified.
- [x] A1 data/source audit, CPU tests and CUDA training/closed-loop smoke passed.
- [x] A1 Stage A completed 3 epochs.
- [x] Complete A1 Stage B3 and C5; same verified data and frozen encoders.
- [x] Seen128/internal-dev select and freeze checkpoint, then full Seen2470 and Unseen2697 evaluation.
- [x] Verify archived A0 complete A3/B3/C5 and full validation artifacts. Produce full A0/A1 reference tables, retrieval, paired/per-map analysis and server artifact inventory; commit/push A1 summaries. Label historical A0 accurately.
- [x] Verify A1 publication_complete.json, service exit status and GPU inactivity before opening A2 execution.

## Phase 2 — A2 only after Phase 1 gate
- [ ] Check out research/a2-waldo-landmark-belief-20261027 in an isolated local checkout; read docs/A2_WALDO_LANDMARK_RELATION_BELIEF_20261027.md and applicable instructions.
- [ ] Audit existing code/data and the old 641-observation/30-reliable-sample bottleneck. Supplement actual CityNav train observations, reliable positive/negative ranking supervision and hard negatives; preserve query/group isolation across train/dev/calibration. No GT in policy inputs.
- [ ] Retain and freeze original A0 including RGB residual/controller, WALDO, BERT and SigLIP2. Train only added box–landmark visual/semantic/geometry relation ranker and bounded local belief update. Invalid or unreliable evidence falls back to A0.
- [ ] Complete data audit, meaningful unit tests and real CUDA smoke before full training.
- [ ] Complete 20 ranking epochs plus 5 local-refinement epochs. Select checkpoint using internal dev only. Independently calibrate belief gates, then freeze all parameters. Do not force activation if unsafe or unsupported.

## Phase 3 — Frozen A0/A2 full closed-loop evaluation
- [ ] Evaluate same full Val-Seen2470 and Val-Unseen2697 episodes/controller; no Unseen tuning or reselection.
- [ ] Report SR/OSR/SPL/NE/nDTW/sDTW, localization recall, corrections/errors/fallback, WALDO candidate coverage, latency, per-map results and paired differences; retain visualizations, logs/checkpoints/TensorBoard locally.
- [ ] Commit/push code and small results summaries to A2 branch; preserve all original A0/A1 artifacts. Deliver A0/A1 and A0/A2 conclusions with full result tables and training analysis, including null/negative results honestly.
