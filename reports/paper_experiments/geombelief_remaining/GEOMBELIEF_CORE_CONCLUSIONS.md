# GeomBelief core experiment conclusions

All core experiments complete. Three matched seeds per map geometry arm; original Correct Geometry reused. New formal-control evaluations use the same runtime after exact historical trajectory parity failed. No original checkpoint/result was overwritten. New visual-input controls are frozen interventions; fair visual-input retraining remains Pending.

## 1. Does correct geometry beat language/name/type alone?

- val_seen R@1, Correct − name_type: +8.876 pp (map-cluster 95% CI [+6.028, +11.776]; positive pointwise map-cluster support).
- val_seen R@1, Correct − shuffled: +7.062 pp (map-cluster 95% CI [+3.867, +10.406]; positive pointwise map-cluster support).
- val_seen SR, Correct − name_type: +9.042 pp (map-cluster 95% CI [+6.187, +11.897]; positive pointwise map-cluster support).
- val_seen SR, Correct − shuffled: +7.341 pp (map-cluster 95% CI [+4.177, +10.676]; positive pointwise map-cluster support).
- val_unseen R@1, Correct − name_type: +19.385 pp (map-cluster 95% CI [+11.407, +26.104]; positive pointwise map-cluster support).
- val_unseen R@1, Correct − shuffled: +16.337 pp (map-cluster 95% CI [+8.299, +22.869]; positive pointwise map-cluster support).
- val_unseen SR, Correct − name_type: +19.095 pp (map-cluster 95% CI [+11.047, +25.955]; positive pointwise map-cluster support).
- val_unseen SR, Correct − shuffled: +16.104 pp (map-cluster 95% CI [+8.009, +22.662]; positive pointwise map-cluster support).

## 2. Which simple spatial relations add value?

- Unseen R@1, distance-minus-name_type: +19.087 pp (map-cluster 95% CI [+12.295, +24.605]; positive pointwise map-cluster support).
- Unseen SR, distance-minus-name_type: +19.095 pp (map-cluster 95% CI [+12.234, +24.653]; positive pointwise map-cluster support).
- Unseen R@1, direction-minus-name_type: +5.860 pp (map-cluster 95% CI [+1.776, +10.248]; positive pointwise map-cluster support).
- Unseen SR, direction-minus-name_type: +5.698 pp (map-cluster 95% CI [+1.491, +10.086]; positive pointwise map-cluster support).
- Unseen R@1, distance_direction-minus-name_type: +18.029 pp (map-cluster 95% CI [+9.298, +25.124]; positive pointwise map-cluster support).
- Unseen SR, distance_direction-minus-name_type: +18.057 pp (map-cluster 95% CI [+9.279, +25.168]; positive pointwise map-cluster support).
- Unseen R@1, distance_direction-minus-distance: -1.058 pp (map-cluster 95% CI [-2.998, +1.161]; CI crosses zero).
- Unseen SR, distance_direction-minus-distance: -1.038 pp (map-cluster 95% CI [-2.955, +0.897]; CI crosses zero).
- Unseen R@1, distance_direction-minus-direction: +12.169 pp (map-cluster 95% CI [+7.522, +15.945]; positive pointwise map-cluster support).
- Unseen SR, distance_direction-minus-direction: +12.359 pp (map-cluster 95% CI [+7.788, +16.063]; positive pointwise map-cluster support).

## 3. Localization, visual correction, trajectory and Stop

- controller_fixed_goal, Unseen SR, oracle_direct-minus-predicted_direct: +77.679 pp (map-cluster 95% CI [+69.192, +86.413]; positive pointwise map-cluster support).
- controller_fixed_goal, Unseen SPL, oracle_direct-minus-predicted_direct: +78.240 pp (map-cluster 95% CI [+69.998, +86.835]; positive pointwise map-cluster support).
- controller, Unseen SR, oracle_direct-minus-predicted_direct: +77.308 pp (map-cluster 95% CI [+68.744, +86.247]; positive pointwise map-cluster support).
- controller, Unseen SPL, oracle_direct-minus-predicted_direct: +77.949 pp (map-cluster 95% CI [+69.682, +86.721]; positive pointwise map-cluster support).
- controller_fixed_goal, Unseen SR, oracle_guarded_blend-minus-predicted_guarded_blend: +77.679 pp (map-cluster 95% CI [+69.192, +86.413]; positive pointwise map-cluster support).
- controller_fixed_goal, Unseen SPL, oracle_guarded_blend-minus-predicted_guarded_blend: +77.480 pp (map-cluster 95% CI [+69.281, +86.050]; positive pointwise map-cluster support).
- controller, Unseen SR, oracle_guarded_blend-minus-predicted_guarded_blend: +77.419 pp (map-cluster 95% CI [+69.192, +86.081]; positive pointwise map-cluster support).
- controller, Unseen SPL, oracle_guarded_blend-minus-predicted_guarded_blend: +77.309 pp (map-cluster 95% CI [+69.361, +85.806]; positive pointwise map-cluster support).
- controller_fixed_goal, Unseen SR, predicted_waypoint-minus-predicted_direct: -2.558 pp (map-cluster 95% CI [-4.703, -0.494]; negative pointwise map-cluster support).
- controller_fixed_goal, Unseen SPL, predicted_waypoint-minus-predicted_direct: -7.681 pp (map-cluster 95% CI [-10.257, -4.500]; negative pointwise map-cluster support).
- controller, Unseen SR, predicted_waypoint-minus-predicted_direct: -2.521 pp (map-cluster 95% CI [-4.703, +0.030]; CI crosses zero).
- controller, Unseen SPL, predicted_waypoint-minus-predicted_direct: -7.723 pp (map-cluster 95% CI [-10.589, -4.186]; negative pointwise map-cluster support).
- controller_fixed_goal, Unseen SR, predicted_guarded_blend-minus-predicted_waypoint: +2.558 pp (map-cluster 95% CI [+0.494, +4.703]; positive pointwise map-cluster support).
- controller_fixed_goal, Unseen SPL, predicted_guarded_blend-minus-predicted_waypoint: +7.598 pp (map-cluster 95% CI [+4.454, +10.144]; positive pointwise map-cluster support).
- controller, Unseen SR, predicted_guarded_blend-minus-predicted_waypoint: +2.410 pp (map-cluster 95% CI [+0.066, +4.538]; positive pointwise map-cluster support).
- controller, Unseen SPL, predicted_guarded_blend-minus-predicted_waypoint: +7.522 pp (map-cluster 95% CI [+4.272, +10.155]; positive pointwise map-cluster support).
- controller_fixed_goal, Unseen SR, predicted_guarded_blend-minus-predicted_direct: +0.000 pp (map-cluster 95% CI [+0.000, +0.000]; CI crosses zero).
- controller_fixed_goal, Unseen SPL, predicted_guarded_blend-minus-predicted_direct: -0.082 pp (map-cluster 95% CI [-0.117, -0.044]; negative pointwise map-cluster support).
- controller, Unseen SR, predicted_guarded_blend-minus-predicted_direct: -0.111 pp (map-cluster 95% CI [-0.418, +0.166]; CI crosses zero).
- controller, Unseen SPL, predicted_guarded_blend-minus-predicted_direct: -0.201 pp (map-cluster 95% CI [-0.500, +0.086]; CI crosses zero).
- controller_fixed_goal, Unseen SR, predicted_guarded_blend-minus-rule_arrival: +0.000 pp (map-cluster 95% CI [+0.000, +0.000]; CI crosses zero).
- controller_fixed_goal, Unseen SPL, predicted_guarded_blend-minus-rule_arrival: -0.000 pp (map-cluster 95% CI [-0.000, +0.000]; CI crosses zero).
- controller, Unseen SR, predicted_guarded_blend-minus-rule_arrival: +0.000 pp (map-cluster 95% CI [+0.000, +0.000]; CI crosses zero).
- controller, Unseen SPL, predicted_guarded_blend-minus-rule_arrival: -0.000 pp (map-cluster 95% CI [-0.000, +0.000]; CI crosses zero).
- Allowing later belief updates, Unseen SR native Guarded-Blend minus initially frozen belief: +0.260 pp (map-cluster 95% CI [+0.000, +0.395]; CI crosses zero).
- Frozen calibrated residual, Unseen SR Real − geometry_pose: +1.335 pp (map-cluster 95% CI [+0.578, +2.010]; positive pointwise map-cluster support).
- Frozen calibrated residual, Unseen SR Real − rgb_text: -0.779 pp (map-cluster 95% CI [-2.486, +0.604]; CI crosses zero).
- Frozen calibrated residual, Unseen SR Real − current_rgb: +1.075 pp (map-cluster 95% CI [+0.000, +2.445]; CI crosses zero).

Oracle intervention quantifies recoverable error under a specified diagnostic controller, not a uniquely additive fraction attributable to localization. OSR−SR describes reaching but not ending in the success region, not automatically a learned-Stop failure. Geometry ablations use a transferred pretrained controller; formal controller controls use the original complete C weights; calibrated visual experiments use the previously trained residual. These different configurations cannot have their gains added together.

## Recommended paper tables and scope of claims

1. Main comparison table: GeomBelief is the paper name of the existing frozen formal GraphNav-V1 C model. Keep the archived official three-split results and exact protocol. Report current-runtime validation deltas separately; do not replace official results with a favorable new run or seed.
2. Core mechanism table: six language/map input schemes, all three seeds, Seen/Unseen R@1/R@4/R@8/NE and navigation SR/SPL/OSR/NE/False-Stop. Include map-cluster intervals for the three predeclared contrasts. The same output grid and network capacity are retained.
3. Controller table: six policies in each of two scopes (native dynamic and identical initial belief), same current runtime, with Oracle rows clearly diagnostic. Use the identical-field scope for execution-only attribution. Keep raw Learned Waypoint, Direct, Guarded-Blend and rule arrival irrespective of sign.
4. Visual table: archived four-arm full validation plus explicitly labeled frozen residual-input interventions. Same visibility gate and same geometric baseline are disclosed. Do not describe masked inference as retrained modality baselines.
5. Qualitative figure: actual language, reference polygons, real visited RGB and executed trajectories, including visual degradation and false stop. Correct/shuffled posterior uses predeclared Seed0/first public-geometry-changed query, not the best-looking seed.

The experiments establish the usefulness (or lack thereof) of supplied map geometry conditional on language and landmark semantics. They do not alone isolate a language–geometry interaction from an unconditional spatial prior, nor distinguish a probabilistic goal belief from every possible point-regression locator. Keep those stronger claims out of the core innovation statement without further controlled evidence. Previous Polygon-vs-BBox results remain unchanged and are not promoted to a new positive claim.

## Statistical and data limitations

Map-cluster bootstrap resamples whole maps; paired episode bootstrap is supplementary. Only four Unseen map blocks from two cities exist. Pointwise intervals are not familywise corrected. Seed means and sample standard deviations are reported; seeds are not extra independent maps. Per-map and leave-one-map-out results are retained. Shuffle effectiveness is measured explicitly because repeated public reference bundles can survive query permutation.

## Remaining experiments

- Fair retraining for Geometry+Pose / RGB+Text / Current-RGB residual inputs: Pending. Existing three-epoch training required 9565.5 seconds per scheme, so approximately 8 GPU-hours for three single-seed budget-matched runs, plus full evaluation. Three seeds would triple that budget. No unfinished row is marked Done.
- External strict official-weight/API replications retain their existing documented blocks; this round refreshed source/protocol evidence and did not train baselines. GeoNav (DeepSeek Flash) remains a substituted API experiment, never GPT-4o.
