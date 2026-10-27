## [core-rgb-pair-20261027] - 2026-10-27
### Features
- Execute matched A0/A1 seed-0 A3/B3/C5 training with TensorBoard, immutable shared initialization checks, and complete Val-Seen/Val-Unseen evaluation.
- Freeze both Seen-selected checkpoint hashes after both training arms finish; report per-map metrics, goal recall, paired bootstrap intervals and exact SR McNemar statistics.
### Design Rationale
- Reuse the fully verified reconstructed 87,512-state cache and immutable epoch-zero snapshot because historical migrated frame/head-cache assets are incomplete. Original A3/head/reference files remain SHA-checked, and bootstrap A3/controller tensors must exactly match them.
- Full-split results cannot change checkpoints. The final paired comparison uses identical original controls, grid, frozen encoders and training inputs.
### Notes & Caveats
- Single training seed; bootstrap episode intervals do not measure training-seed uncertainty. Dataset duplicates are retained; SR also reports query-cluster intervals.
- Goal recall deduplicates map+instruction and distinguishes initial static, first observed and final observed belief. False-Stop uses policy stops as denominator, with episode fraction reported separately.
- Checkpoints, TensorBoard events and raw trajectories remain on the local server under runs/joint_rgb_geometry_core_20261027.
