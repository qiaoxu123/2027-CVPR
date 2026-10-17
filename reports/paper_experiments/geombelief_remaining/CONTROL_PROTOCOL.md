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
