# Polygon-to-RGB projection-only audit

Scope supersedes the proposed relation training: no model modification/training, no relationship inference, no navigation or threshold changes.

1. Select 48 TRAIN public named Polygon/view pairs (source fragments may recur across strata, count unique fragments separately), Building/Road × full/partial/oversized/outside, with varied recorded TRAIN XY and diagnostic yaw/height. All source contours are complete; no target GT or target-nearest selection. Diagnostic re-renders are explicitly not claimed as original human trajectory frames.
2. Use the unmodified native384 renderer, record actual pose/Footprint/RGB SHA. Compare advertised homography with the actual raster-index integer warp, including the pixel-center convention. Small roundtrip alone is not registration accuracy.
3. Compute exact Polygon/view intersection, image occupancy, RGB/depth support masks, rasterization differences, actual 50%/90% ROI acceptance and the separate native-patch/weak-label 90% effects. Do not change production preprocessing/thresholds.
4. Provide 48 original PNGs, projected outlines, geometric clipped masks, effective-source masks, HTML and editable human review JSON. Direct assistant visual inspection is labeled AI review, never human GT. No pixel instance IoU without verified human masks.
5. Run coordinate/crop/border/threshold unit tests, preserve original/source SHA, report numeric vs semantic limitations, commit/push current PGVB branch. Only decide whether further calibration evidence is needed; do not presume a learned registration network necessary.

Sampling clarification: oversized means source Polygon area >1.25×view area with ≥10% view overlap; it need not contain 90% of the camera view. Within-stratum source polygons are distinct; cross-stratum repeated landmarks provide scale/pose comparisons.
