# GeomBelief visual mechanism audit

Audit date: 2026-10-20. **The formal main model and the separately trained Calibrated Visual Residual (CVR) are different policies.** This audit changes neither implementation nor weights. Source links below pin experiment commit `16452896e8b169a9c540ce6de7f7d00b17b453a7`; the inspected source files match the archived inference implementation hashes (see `CODE_PROVENANCE.json`).

## 1. Identity and actual entrypoints

|Policy|Identity|Inference|
|---|---|---|
|Formal GeomBelief / GraphNav-V1 C epoch1|Base SHA `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`|`scripts/evaluate_closedloop.py --model graphnav-v1`; `GraphNavStage1Policy`|
|Independent CVR|Same frozen base plus head SHA `5fc1a3e115b71d14fe1e8fbaf3ed29c1863d8b2bc080455d89f7a4e503d25f11`|Same policy, with `install_calibrated_residual(policy, head, base)`|

Formal configuration: 28×28 candidates, frozen BERT and SigLIP2 Large/384 (revision `1b426889ea62b5a72bf9839009a1b184bfc9c178`), history enabled, residual bound 0.5, no active observation/new GAT/relation loss, guarded-blend, 20 decisions, XY≤50m and |Δz|≤10m, Stop threshold 0.3, positive weight 20. Exact archived protocol is in `CODE_PROVENANCE.json` and `visual_evaluations/calibrated_residual_full_unseen_real/metrics.json`.

The concrete CLI switch for CVR is `--stage1-residual-checkpoint <head.pt>`; `--checkpoint <base.pt> --model graphnav-v1 --stage1-visual-control {real,zero,shuffle,no-belief} --stage1-history-control checkpoint` select the base/control. `scripts/evaluate_closedloop.py:219–225` constructs the policy and observation builder; lines270–276 install/check the frozen residual. The archived `metrics.json:args` records the full invocation. Unseen CVR evaluation requires the pre-frozen validation manifest; this audit did not bypass that check or run the command. The parser's residual-switch help string still says Seen-only, while its actual validation branch permits Seen and pre-frozen Unseen; do not treat that help string as the executed protocol.

**Available evidence:** archived per-step tensors, hashes, replay checks and exact recoverable image crops. The actual two neural checkpoint files are not present at their original `/home/rental/...` paths on this host; no new full neural replay is claimed. Q=784 is measured from arrays. The actual controller K is the checkpoint's `a3_args.top_k` (source default 20); it must not be inferred from the diagnostic Top-16 export when the checkpoint itself is unavailable.

## 2. RGB and candidate association

1. [`NativeObservationBuilder.build_from_poses`, dynamic_runtime.py:L20-L30](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/dynamic_runtime.py#L20-L30) passes **only the last executed pose** to the environment renderer. The policy never receives the entire source raster.
2. [`CityNavObservedViewRenderer.current_rgbd`, renderer.py:L55-L86](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/renderer.py#L55-L86) produces BGR `[384,384,3]`, a depth-valid mask, a vertical depth proxy and four world-XY corners. RGB is an orthophoto homography crop, **not onboard perspective imagery or a 3D occlusion renderer**. Depth is camera Z minus published surface height. Support also rejects black RGB, insufficient bilinear source support, invalid height and height≥camera Z.
3. [`ObservedRGBMemory.view_corners`, memory.py:L43-L57](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/memory.py#L43-L57) makes a square of side `2*(camera_z−map_GL)`, oriented by yaw. There is no measured pitch/roll or calibrated physical perspective intrinsics. An equivalent flat-plane 90° footprint is not a validation of a real pinhole camera.
4. [`world_to_frame` / `frame_to_world`, landmark_projection.py:L30-L54](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/landmark_projection.py#L30-L54) map candidate world XY and metric square ROIs to the observed frame using the **four-corner homography**. Association is coordinate-based; it is not semantic detection, SIFT matching, learned correspondence, or a nearest-GT-instance lookup.
5. [`ObservedPatchScaleEncoder.encode`, observed_patch_scales.py:L150-L180](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/observed_patch_scales.py#L150-L180) runs the frozen SigLIP2 vision transformer on the whole current crop, yielding 24×24=576 contextual patch tokens of dimension 1024. Candidate-centered 32/64/128m squares select tokens; native attention pooling produces three `[1024]` vectors per candidate.

This is **shared whole-frame contextual token pooling**, not three independently re-encoded crops or higher-resolution images. Selected patches already contain transformer context from the current frame. Consequently a candidate ROI is not a certified isolated view of its target object.

### Actual ROI-validity test

[`_small_grid_masks`, observed_patch_scales.py:L14-L49](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/observed_patch_scales.py#L14-L49) is the executed Q=784 path. For candidate q and scale s:

- all four ROI corners must project into `[0,383]²`;
- `sqrt(projected_polygon_area) >= 24 px`;
- at least 90% of ROI pixels must be nonblack and depth-valid;
- at least one selected 16×16 patch has ≥90% valid support.

`masks:[Q,3,576]`, `roi_valid=masks.any(-1):[Q,3]`, `roi_quality:[Q,3,3]`. The three qualities are projected span/384, valid-pixel fraction, and metric extent/128. Invalid feature vectors are zero, with an independent valid bit. **This mask tests usable ROI support, not target semantic identity or true optical visibility.** Large-grid optimizations for Q>8192 are separate and are not needed to explain the formal 784-cell run.

## 3. Tensor trace

Batch B=1 in rollout, Q=784, S=3, visual D=1024, dense context H=128. N and K come from the frozen checkpoint configuration; L≤96.

|Stage|Tensor|
|---|---|
|BERT instruction / landmark names|`tokens [B,L,768]`, token mask `[B,L]`, names `[B,N,768]`|
|Relative candidate–landmark geometry|`pair_geometry [B,Q,N,10]`|
|Static A3|`s0 [B,Q]`, `dense_context [B,Q,128]`|
|Current image encoder|patch tokens `[B,576,1024]`|
|Current/history candidate features|`[B,Q,3,1024]` each|
|Current/history support|`[B,Q,3]` boolean each|
|Current/history ROI quality|`[B,Q,3,3]` each|
|Appearance text|normalized native full-instruction + `appearance_query` text, `[B,1024]`; not a BERT-to-SigLIP projection|
|Current/history matcher score|`[B,Q]` each|
|History fusion gate and formal Δ|`[B,Q]` each|
|CVR input MLP|`[B,Q,106] → [B,Q,64] → [B,Q,2]`|
|Updated belief|logits `[B,Q]`, probabilities `[B,Q]`|
|Controller proposals|XY `[B,K,2]`, context `[B,K,128]`, gathered scores `[B,K]`|
|Controller output|paths `[B,K,3,8,3]`, modes `[B,K,3]`, ranking/Stop `[B,K]`|

The two CVR output channels are raw correction and a confidence-gate logit; they are not two semantic class probabilities.

## 4. Actual static and formal visual formulas

### Static model

[`FullTokenGeometryLocalizer.forward`, full_localization.py:L105-L188](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/full_localization.py#L105-L188):

\[
s_0(q)=\mathrm{prior}(q)+w_r\mathrm{rules}(q)
 +\tanh(g_T)T(q)+\tanh(g_G)G(q).
\]

T uses candidate positional/rule features and cross-attention to BERT tokens. G attends candidate-relative geometry plus landmark name/type/shape embeddings, including a learned null anchor. In the formal `variant='geometry'`, name/type/node projections are used **without inter-node GAT message passing**. The manuscript's four-term formula is a legitimate abstraction only if gains and rule normalization remain explicit.

### Current appearance and history

[`MultiScaleAppearanceMatcher.forward`, appearance_matching.py:L30-L62](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/appearance_matching.py#L30-L62): adapted normalized image/text embeddings use residual MLPs at coefficient 0.1. Scale weights are a learned quality softmax restricted to valid scales, renormalized after masking. The score is their weighted cosine sum times `min(exp(log_scale),100)`.

[`QualityHistory.forward`, stage1.py:L23-L32](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/stage1.py#L23-L32) computes a sigmoid gate from concatenated mean current/past quality (six numbers per candidate), with missing-view overrides. Let c(q), h(q) be the two matcher scores and w(q) the overridden gate:

\[
a_t(q)=w(q)c(q)+(1-w(q))h(q),\quad
m_t^F(q)=\bigvee_s v_{tqs}\;\lor\;\bigvee_s v^{hist}_{tqs}.
\]

Without history, a=c and m is current support only. **Yes, there is a real per-candidate valid observation mask.** It is `observed` in `GraphNavStage1.forward`, not a merely theoretical symbol.

[`GraphNavStage1.forward`, stage1.py:L107-L125](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/stage1.py#L107-L125):

\[
\bar a_t=\frac{\sum_qm_t^F(q)a_t(q)}{\max(1,\sum_qm_t^F(q))},\qquad
\Delta_t^F(q)=0.5\,\tanh(g_v)\tanh((a_t(q)-\bar a_t)/5)\,m_t^F(q).
\]
\[
s_t=s_0+\Delta_t^F,\qquad B_t=\mathrm{softmax}_q(s_t).
\]

This is bounded tanh scaling and observed-score centering, not a hard clip of a raw residual or a Bayesian posterior update. Unknown candidates receive zero direct Δ; normalization can still change their probabilities. Fallback keeps the original proposal tie order/dtype to avoid changing behavior solely through BF16→FP32 casting.

## 5. Independent CVR formula and additional confounds

[`CalibratedVisualResidual.forward`, calibrated_visual_residual.py:L21-L48](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/calibrated_visual_residual.py#L21-L48):

\[
m_t^C(q)=m_t^F(q)\land
\{\|F^{cur}_{q,:,:}\|_1+\|F^{hist}_{q,:,:}\|_1>0\}.
\]

Scales are pooled by normalized `clamp(quality_span,0,1)*clamp(quality_fraction,0,1)*valid`. The MLP input concatenates:

1. frozen dense geometric/language context projected 128→16;
2. current and history image projections 1024→32, each multiplied elementwise by the projected normalized native text (32+32 dimensions);
3. both quality tensors flattened (9+9 dimensions);
4. eight scalar features: max-relative/clamped static logit, static probability×Q/784, original current/combined appearance scores divided by20, current support, history support, normalized candidate-to-UAV XY distance (`use_pose=True` default), final known flag.

Let r(q), z(q) be the two MLP outputs. For the selected centered/default-pose model:

\[
\bar r=\frac{\sum_q m_t^C(q)r(q)}{\max(1,\sum_q m_t^C(q))},\qquad
\Delta_t^C(q)=2\tanh(r(q)-\bar r)\sigma(z(q))m_t^C(q).
\]

Evidence strength for configuration: per-step diagnostics explicitly record `residual_bound=2.0` and `residual_centered=True`. The archived selected `calibrated_residual/config.json` predates the later `model_kwargs` flags; the loader uses constructor defaults (`use_pose=True`) when this dictionary is absent. Its expected pose branch is described above. The actual serialized checkpoint kwargs cannot be independently reopened on this host. The later optional `use_pose=False` branch replaces the distance scalar with current-image/native-text cosine; it does not remove geometry, ROI quality or support. Do not claim this later variant was used in the selected run.
\[
s_t^C=s_0+\Delta_t^C\quad\text{(not }s_0+\Delta_t^F+\Delta_t^C\text{)}.
\]

[`install_calibrated_residual`, calibrated_residual_runtime.py:L15-L40](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/calibrated_residual_runtime.py#L15-L40) first calls the formal forward with `no-belief` (old matcher/history scores still computed), then **replaces** the residual and belief. Formal visual, history, geometric and controller weights remain frozen.

`sigmoid(z)` is a learned attenuation gate supervised indirectly through goal-mass loss. No saved ECE, reliability curve, calibrated probability target, or other evidence establishes empirical confidence calibration. Keep “Calibrated Visual Residual” as a model name if needed, but describe the implemented operation as **confidence-gated**, not demonstrated probabilistic calibration.

Important export detail: the wrapper updates `base.delta` and `base.belief` but does **not** replace `base.observed`. Thus cached `*_observed` is the original formal support flag, not necessarily mC (especially under Zero). `corrected_field.visual_coverage=(delta!=0)` is an applied-residual indicator, also not a semantic visibility mask. The CVR confidence output itself was not cached in the selected replay arrays and is not invented here.

## 6. History storage and update order

[`CandidateMemory`, stage1_data.py:L12-L36](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/stage1_data.py#L12-L36) stores per-candidate/per-scale feature sums `[Q,3,1024]`, weights `[Q,3]`, and quality sums `[Q,3,3]`. Accepted views add quality-weighted vectors. `tensors()` returns their averages, validity and averaged qualities. This is neither a stack of raw historical RGB images nor temporal cross-attention over all images.

The step must increase strictly. Views with no supported ROI are skipped. A view whose XYZ is within2m of the **last accepted** pose is skipped; yaw is not used by this duplicate test. This is not global loop-closure or complete repeated-view removal.

[`GraphNavStage1Policy.predict`, stage1_runtime.py:L54-L75](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/stage1_runtime.py#L54-L75) reads memory **before** predicting and only updates it afterward. At t=0 memory is empty. The cached static field is computed once per episode. Each step uses that fixed s0 and current/past feature evidence; it does **not** receive Bt−1 as the next prior. Feature memory is recurrent state; the goal-belief logits are not recursively accumulated.

## 7. Real / Zero / Shuffle / No-belief

|Control|Formal main model|Independent CVR|
|---|---|---|
|Real|Original current/history ROI vectors, masks and qualities|Same inputs; learned CVR added to fixed static field|
|Zero|Zero current **and past feature tensors** inside forward; preserve masks, qualities and language. Learned adapter biases/quality may still yield a score, so this is not definitionally equivalent to No-belief|Wrapper zeros both feature tensors. CVR nonzero-feature gate then makes mC false and ΔC=0; this also changes the content-dependent gate|
|Shuffle|Permute supported candidate vectors **within each scale**, independently for current/history using fixed seed0; leave receiver validity/quality/coordinates unchanged|Wrapper applies the same supported-vector intervention before formal matcher and CVR. It does not shuffle text, pose, quality, candidate coordinates or the full trajectory|
|No-belief|Force ΔF=0 and return static field; encoders/appearance/history can still execute|Skip added head correction, ΔC=0; formal old visual residual remains disabled too|

Raw unperturbed current ROI vectors are what runtime subsequently stores in memory. Each control is applied again to its forward inputs; it does not rewrite archived images or retroactively change the real path in matched-path replay.

**Semantic inference limit:** CVR has pose-distance, geometric context, prior scores, ROI quality and support inputs. Real-vs-Zero removes content and changes the meaningful-feature gate; Real-vs-Shuffle tests image-to-location association and can confound semantics with local texture/geometry/view structure. Neither alone establishes semantic object recognition. In the selected correction case, Shuffle also succeeds; in the degradation case, Shuffle also fails.

## 8. Trajectory and Stop: actual executed branch

Both policies control using the **updated** belief. Runtime calls `A3TrajectoryPolicy.predict_field(state, output.belief)` after the wrapper. That function recomputes the frozen/current controller outputs from the supplied field, using updated proposal order, gathered updated logits, normalized global entropy, current pose, public XY bounds, global GL and static dense context selected at the new IDs. `observed_fraction` is explicitly zero.

Crucial distinction: `GraphNavStage1.forward` also computes an `output.trajectory`, but runtime does not execute that returned object. The CVR wrapper leaves that object unchanged; execution remains correct for the new belief because `predict_field` recomputes the head. Do not use that stale forward-returned trajectory as the actual post-CVR command in a figure.

[`A3TrajectoryPolicy.predict_field`, a3_runtime.py:L121-L164](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/a3_runtime.py#L121-L164): guarded-blend (`explore_candidates=0`) selects updated-belief candidate index0, then that candidate's best mode. The trained candidate-ranking head does **not** choose a different goal in this deployed branch. Context is the static128-D context, not a newly computed visual latent. No local raster elevation field is fed to this controller; only the map GL scalar and pose altitude are included.

[`CandidateTrajectoryStop.forward`, trajectory_policy.py:L86-L134](https://github.com/qiaoxu123/2027-CVPR/blob/16452896e8b169a9c540ce6de7f7d00b17b453a7/graphnav/trajectory_policy.py#L86-L134) produces three modes×eight waypoints. Runtime executes one bounded first waypoint, with learned deviation tapered near the predicted goal and blend≤0.25.

\[
p_{stop}=\sigma(z_{stop}-\ln20),\quad
\mathrm{Stop}=\{p_{stop}\ge .3\}\land\{d(\mathrm{pose},\widehat g)\le .001m\}
\land\{10m\le z-GL\le100m\}.
\]

The odds-shift correction is analytic weighted-BCE compensation, not proof of empirically calibrated Stop probabilities. The 20m evaluator radius is different from the 1mm predicted-goal guard. Wrong-goal stopping can satisfy every controller guard.

## 9. Recorded training objectives and what is not implemented

Formal `[A3,B3,C5]`, selected C1: stageA/B freeze A3/controller, stageC unfreezes their active branches; history is trainable only in B/C and inactive GAT/old-vision branches remain frozen. BERT/SigLIP remain frozen. `stage1_losses` at stage1.py:L128-L151 uses

\[
L_{A/B}=L_{static}+L_{belief}+.1L_{visual}+.05L_{hardrank},
\]
\[
L_C=L_{A/B}+.1L_{path}+.02L_{mode}+.05L_{navrank}+.05L_{stop}+L_{anchor}.
\]

Stop BCE uses positive weight20. Targets and expert future waypoints enter loss-only functions. The four-term Lnav in the manuscript can summarize a controller sub-objective, but **cannot be presented as the full stageA/B/C training loss**.

Independent CVR: three recorded epochs,83,140 training states/4,372 internal dev states; choose internal-dev R1 then mass loss. Freeze original A3, matcher, quality history, controller and foundation encoders. With default rank_weight=0, `correction_loss` at calibrated_visual_residual.py:L61-L75 is legal20m mass NLL +0.1 non-improvement penalty +0.02 supported residual-square drift. Optional later CLI contrast/rank flags are not evidence that they were enabled in this selected run.

Not established/implemented in this policy: true perspective/optical occlusion rendering, target-instance visibility labels, semantic feature-to-map matching independent of supplied pose, raw-RGB temporal attention, recursive Bayesian belief fusion, active information-gain planning, formal-mode inter-landmark GAT propagation, visual latent context for the controller, or empirical CVR confidence calibration. Abstract paper concepts must not imply these exist.

## 10. Numerical consistency

`CAUSAL_SUMMARY.json` / `VISUAL_CAUSAL_CASES.csv` independently recompute all310 original comparisons from saved tensors. Unobserved recorded support receives exactly zero direct residual. Maximum saved-logit discrepancy from s0+Δ is recorded (float capture rounding, below1e−5). Entropy is computed in nats from the same normalized fields. Candidate-mask presence, vector input sensitivity, changed Top-1, semantic understanding, and navigation success remain separate statements.
