# GeomBelief frozen complete-model control attribution

Protocol record. Actual completion and results are reported per scope/split below.

Complete formal C Epoch1 SHA: 00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a. All language, visual and trajectory/Stop parameters remain frozen. Original full Seen/Unseen Guarded-Blend is retained as historical reference. A separate same-runtime Guarded-Blend is evaluated after smoke parity found small intermediate-pose numerical drift; controller paired inference uses the new matched runtime control. This is distinct from the map-ablation's legacy controller transfer.

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

## Strict identical-goal control

The native dynamic scope fixes weights and candidate-selection rules, but different motion yields different future RGB and thus different later candidate fields. It cannot alone satisfy identical candidate values across controllers. Therefore the primary execution-only analysis also uses **identical_initial_belief**: all six controls consume the exact same serialized first-observation GoalField (logits, candidate IDs/coordinates and candidate contexts) for an Episode. Only pose and the chosen execution/Stop intervention vary; Oracle overrides the first controller coordinate only. All first fields are produced by the original complete C weights from a real initial RGB observation, before using any target or teacher data. No future observation is available to these controllers.

This scope is a controlled diagnostic, not the native dynamic formal policy. Guarded-Blend refers to the unchanged execution strategy under a frozen initial belief. Its timing excludes precomputation and must not be presented as native full-model profiling. Every trace has the shared initial_field_sha256; field equality across arms is checked. Both Seen128 and Unseen128 precede the full split. Native dynamic-minus-initial-frozen Guarded-Blend measures allowing later belief updates, with coupled observation/path consequences, not an additive isolated RGB percentage.

## Runtime and recovery evidence

Historical exact trajectory parity was rejected (birmingham_block_9:110:0; same final NE/SR but small intermediate-pose differences). A new matched-runtime formal control is retained separately; original published metrics remain unchanged. The attempted prefix replay was not used. See rule_replay_certificate.json.

Parallel scratch-file collision was detected and repaired by per-process input/render directories and per-arm locks. All potentially affected tails were quarantined, with the verified pre-parallel prefix retained. See scratch_isolation_recovery.json; excluded output is never part of metrics. This scheduling repair changes no policy weights or parameters.

Metric definitions: SR is final XY NE ≤20 m; OSR is any visited pose within 20 m; SPL uses the shared evaluator’s original shortest-distance/path-length convention. False-Stop Rate is failed active policy stops divided by all active policy stops, and is undefined (`null`) when there are no active stops. The per-Episode false-stop indicator is also retained separately; it has a different denominator.

## Arrival arithmetic audit

The shared controlled-policy wrapper recomputes the nominal 0.001 m world-distance gate using NumPy float64, whereas the archived runtime computed its norm in float32 tensors. Offline GPU float32-norm checks at every recorded decision in all 24 full-split control runs found zero arrival-boolean disagreements and zero Stop-request disagreements attributable to this precision difference. See arrival_precision_audit.json and scripts/geombelief_arrival_precision_audit.py. This is an audit of observed states, not a counterfactual replay or a reason to overwrite archived formal metrics.

## Results

|scope|arm|split|status|episodes|SR|SPL|OSR|NE|False_Stop_Rate|failed_without_visited_success_region|failed_with_visited_success_region|
|---|---|---|---|---|---|---|---|---|---|---|---|
|native_dynamic_belief|predicted_guarded_blend|val_seen|Done|2470|0.4761133603238866|0.46002843520078085|0.5566801619433198|29.11917144218048|0.5238866396761134|1095|199|
|native_dynamic_belief|predicted_direct|val_seen|Done|2470|0.4757085020242915|0.46230986248381456|0.551417004048583|29.16107078935657|0.5220618556701031|1108|187|
|native_dynamic_belief|predicted_waypoint|val_seen|Done|2470|0.38785425101214577|0.27557750557373967|0.6631578947368421|32.25687156832064|None|832|680|
|native_dynamic_belief|oracle_direct|val_seen|Done|2470|1.0|0.9999999998027225|1.0|1.359082426619433e-07|0.0|0|0|
|native_dynamic_belief|oracle_guarded_blend|val_seen|Done|2470|1.0|0.9920129553640376|1.0|0.00024830131855792687|0.0|0|0|
|native_dynamic_belief|rule_arrival|val_seen|Done|2470|0.4761133603238866|0.46002843632399987|0.5566801619433198|29.117037333265248|0.5238866396761134|1095|199|
|native_dynamic_belief|predicted_guarded_blend|val_unseen|Done|2697|0.22580645161290322|0.21849332847569689|0.3170189098998888|51.53145623880075|0.7740259740259741|1842|246|
|native_dynamic_belief|predicted_direct|val_unseen|Done|2697|0.22691879866518352|0.22050504464804493|0.30626622172784573|51.55983731628355|0.7721614485099962|1871|214|
|native_dynamic_belief|predicted_waypoint|val_unseen|Done|2697|0.20170559881349648|0.14327043573497775|0.40526510938079346|53.66373651051866|None|1604|549|
|native_dynamic_belief|oracle_direct|val_unseen|Done|2697|1.0|0.9999999998340132|1.0|1.1598263840609937e-07|0.0|0|0|
|native_dynamic_belief|oracle_guarded_blend|val_unseen|Done|2697|1.0|0.9915826309814292|1.0|0.0002539481018284038|0.0|0|0|
|native_dynamic_belief|rule_arrival|val_unseen|Done|2697|0.22580645161290322|0.21849332967396487|0.3170189098998888|51.531456018761546|0.7741097922848664|1842|246|
|identical_initial_belief|predicted_guarded_blend|val_seen|Done|2470|0.4744939271255061|0.46048579125539996|0.5534412955465587|29.242682705954667|0.5255060728744939|1103|195|
|identical_initial_belief|predicted_direct|val_seen|Done|2470|0.4744939271255061|0.4622327590435084|0.548582995951417|29.242683715057908|0.5222588623248146|1115|183|
|identical_initial_belief|predicted_waypoint|val_seen|Done|2470|0.3963562753036437|0.27938613283288066|0.6668016194331984|32.20738002122278|None|823|668|
|identical_initial_belief|oracle_direct|val_seen|Done|2470|1.0|0.9999999998027225|1.0|1.359082426619433e-07|0.0|0|0|
|identical_initial_belief|oracle_guarded_blend|val_seen|Done|2470|1.0|0.9919704665107213|1.0|0.0002506778421731737|0.0|0|0|
|identical_initial_belief|rule_arrival|val_seen|Done|2470|0.4744939271255061|0.4604857923787132|0.5534412955465587|29.24268282526683|0.5255060728744939|1103|195|
|identical_initial_belief|predicted_guarded_blend|val_unseen|Done|2697|0.2232109751575825|0.21677632383609471|0.3133110863922877|51.79535111691069|0.7766233766233767|1852|243|
|identical_initial_belief|predicted_direct|val_unseen|Done|2697|0.2232109751575825|0.21759898510057527|0.3007044864664442|51.79534109811899|0.7762264150943397|1886|209|
|identical_initial_belief|predicted_waypoint|val_unseen|Done|2697|0.19762699295513533|0.1407936997417905|0.40378197997775306|53.796477211505895|None|1608|556|
|identical_initial_belief|oracle_direct|val_unseen|Done|2697|1.0|0.9999999998340132|1.0|1.1598263840609937e-07|0.0|0|0|
|identical_initial_belief|oracle_guarded_blend|val_unseen|Done|2697|1.0|0.9915724830792093|1.0|0.0002565398674552991|0.0|0|0|
|identical_initial_belief|rule_arrival|val_unseen|Done|2697|0.2232109751575825|0.21677632391962415|0.3133110863922877|51.79535043201316|0.7767890248424175|1852|243|

Full-split paired controller intervals are in paired_statistics.json. Oracle gains measure recoverable error under the specified intervention; they are not an additive percentage of failure uniquely caused by localization. OSR−SR identifies visited-but-not-retained success, not automatically a Stop error. Learned Stop versus rule uses the same 0.001 m gate.
