# Map-guided visual evidence experiment

Base f2faec2ef0089dd98049ff11ece51df0cc905143. No training, no formal validation/Test, no paused jobs resumed. Independent branch experiment/map-guided-visual-grounding.

1. Audit source rendering, calibration availability, cache association, CityRefer instances. Save stage1 report/commit.
2. Unit-test current-frame polygon projection, partial/outside/support states, exact source proposal alignment, no-GT inference and fixed-region feature correspondence permutation. TRAIN fit/calibration groups only for implementation and constants. Render TRAIN success/failure cases, save stage2 report/commit.
3. Freeze scoring and source hashes before any internal Dev outcome. Evaluate once on635 states using same cached current frames, same15 proposal regions, frozen Native/AerialVG. Map-only uses public geometry and A0 on shared proposals; Map+RGB adds target appearance evidence and frozen AerialVG reference-crop evidence. Shuffled permutes frozen visual-feature/ROI correspondence within current frame only. All methods use same A0/support/fusion geometry; no Gate or network training.
4. Instance evaluation isolated after inference: official TRAIN source object_id and contour -> registered image bbox; IoU >=0.5 top1/top5/top15 on sufficiently large, supported projected instance contours. This is not semantic segmentation. Human error/physical occlusion unknown until independent manual labels.
5. Report total and geometric-visibility strata, projection/ROI yield, registered-instance matching, R1/NE/corrections/harm, measured incremental time/CUDA allocation, success/failure figures, limitations and whether to train. Save stage3/final commit and push every stage.

Protocol limitations: existing RGB is observed CityNav orthographic crop, not a physical perspective camera. footprint_world gives TL/TR/BR/BL; cache supplies depth_valid only, no metric depth, K, camera extrinsics or occlusion truth. Use current footprint homography and depth support; do not invent height correction. Public polygon height can flag uncertainty, not recover missing occlusion. No full orthophoto is passed to inference; map figures contain vector geometry only.

Independent inference API must not accept target IDs/positions/contours; those are evaluation-only. Public landmark catalog uses named objects only (name/geometry/type), never hidden target identity/descriptions. No future frames or global source RGB are read. Current-only evidence is a valid causal subset; do not claim history fusion implemented.
