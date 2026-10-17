# GeomBelief frozen complete-model control attribution

Status: Running. No new outcome is claimed until complete paired split results exist.

Complete formal C Epoch1 SHA: 00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a. All language, visual and trajectory/Stop parameters remain frozen. Original full Seen/Unseen Guarded-Blend is reused. This is distinct from the map-ablation's legacy controller transfer.

|Condition|Waypoint|Stop|Goal|
|---|---|---|---|
|Predicted Direct|Direct to belief Top1, original bounded actuator|Learned Stop ≥0.3 AND arrival ≤0.001 m AND legal altitude|Original dynamic Top1|
|Predicted Learned Waypoint|Raw first waypoint of highest-probability mode, fixed Top1|Same learned+arrival guard|Original dynamic Top1|
|Predicted Guarded-Blend|Original conservative blend=0.25|Original learned+arrival guard|Original dynamic Top1|
|Oracle Direct|Direct to GT, diagnostic only|Same learned+arrival guard|Only controller's Top1 coordinate overwritten|
|Oracle Guarded-Blend|Original blend and same pretrained head|Same learned+arrival guard|Only controller's Top1 coordinate overwritten|
|Rule Arrival|Original Guarded-Blend|Arrival ≤0.001 m AND legal altitude, learned probability ignored|Original dynamic Top1|

Oracle coordinates enter the diagnostic controller only after the unchanged geometry/visual prediction, never the locator or visual network. Candidate context remains the predicted Top1 context, so Oracle+head is a coordinate-intervention diagnostic, not a retrained oracle-context model. No oracle result is a deployable baseline.

All arms use the same candidate-selection rule and network weights, but changed trajectories produce different causal RGB observations and hence may change later beliefs. It is impossible to hold observations identical in genuine closed loop while changing motion. This mediated effect is explicitly part of policy performance, not evidence that the weights changed.

Success uses final XY NE ≤20 m, independently of whether policy_stop occurred. Stop arrival guard uses distance to predicted goal ≤0.001 m. The rule control preserves this guard to isolate the learned Stop decision; it is not an optimized 20 m arrival-radius policy. Maximum 20 decisions, XY step ≤50 m, Z change ≤10 m. Seen128 and Unseen128 precede each split's full evaluation; no tuning follows their outcomes.

Each trace stores initial/final goal error, predicted-goal arrivals, GT success-region visits (post-rollout only), learned Stop request, policy stop, horizon and stagnation. Failure attribution is descriptive/counterfactual, not an additive decomposition of interacting locator/vision/controller effects.

## Results

|arm|split|status|episodes|SR|SPL|OSR|NE|False_Stop_Rate|failed_without_visited_success_region|failed_with_visited_success_region|
|---|---|---|---|---|---|---|---|---|---|---|
|predicted_guarded_blend|val_seen|Done|2470|0.4761133603238866|0.46002843520078085|0.5566801619433198|29.11917144218048|0.5238866396761134|1095|199|
|predicted_direct|val_seen|Pending|||||||||
|predicted_waypoint|val_seen|Pending|||||||||
|oracle_direct|val_seen|Pending|||||||||
|oracle_guarded_blend|val_seen|Pending|||||||||
|rule_arrival|val_seen|Pending|||||||||
|predicted_guarded_blend|val_unseen|Pending|||||||||
|predicted_direct|val_unseen|Pending|||||||||
|predicted_waypoint|val_unseen|Pending|||||||||
|oracle_direct|val_unseen|Pending|||||||||
|oracle_guarded_blend|val_unseen|Pending|||||||||
|rule_arrival|val_unseen|Pending|||||||||

Full-split paired controller intervals are in paired_statistics.json. Oracle gains measure recoverable error under the specified intervention; they are not an additive percentage of failure uniquely caused by localization. OSR−SR identifies visited-but-not-retained success, not automatically a Stop error. Learned Stop versus rule uses the same 0.001 m gate.
