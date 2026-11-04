# Local Anchor utilization and one PGVB retraining

Base: 5175cce, current PGVB branch. Original weights/default navigation are preserved.

1. Add public same-name Polygon union/Footprint intersection, geometric and RGB/depth-supported masks separately. Whole-landmark fraction is metadata, never a local-support gate. Preserve .5m boundary uncertainty and 90% reliable patch support. Verify on prior48 and all existing TRAIN frames, not formal Val/Test.
2. Version new cache/loss/trainer, leaving old frozen artifacts/source reproducible. Keep old selected queries, extend with reliable >=2-positive regions even when same-frame negative count is zero. Unknown windows are never negatives. Zero-negative rows use positive evidence and within-positive uniform consistency, not artificial contrastive negatives.
3. Use exactly the existing 226306-parameter PGVB architecture. Encode clipped local Polygon in actual image coordinates (same 16x4 token shape, absolute map registration is public input), retain legacy full-centered tokens for paired baseline. No new network, relation branch or fusion.
4. Regenerate TRAIN cache from original2324 frame metadata; split fit Cambridge/dev Birmingham as before. Record all source/label/selection transitions and preserve all legacy cohorts. Same-image negatives require mutual known-negative support, not just nonoverlapping positives.
5. After geometry/data checks pass, GPU smoke+resume and one seed0 retraining from scratch. Frozen SigLIP2, same lr/loss weights, upper budget5112 updates (old pilot), up to12 epochs, selection by TRAIN mean loss only. Dev is previously exposed regression, not a new blind confirmation; no tuning/retraining from Dev.
6. Evaluate both checkpoints on identical common old Dev and expanded Dev; report same-image proxy discrimination, R20/NE, zero/shuffle RGB and no-Polygon controls. Also old checkpoint with new local tokens to separate input change from training.
7. Save real RGB/local Anchor/Belief examples, old/new sample counts, negative findings, GPU/config/checkpoints and Chinese report. Test, commit and push. No navigation/geometry fusion/direct or indirect relation inference.

Before Dev scoring: add a frozen wrong-RGB content correspondence control using a different real, fully valid frame from the same subset/different map. Keep target footprint, valid mask, registered Polygon tokens and query unchanged. This checks whether registered-coordinate decoding through positional image embeddings can masquerade as semantic visual matching; no new training or tuning.

Wrong-RGB preflight: no fully-valid replacement exists for some same-map frames. Such diagnostic pairs are excluded only from this control, with its denominator reported and real predictions compared on the same subset; primary old/new Dev evaluation remains complete. No metric predictions existed at the failed preflight, and no model retraining or parameter change followed it.
