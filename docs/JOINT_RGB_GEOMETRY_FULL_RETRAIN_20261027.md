# Joint RGB–Geometry Full Retraining Ablation (2026-10-27)

Status: **protocol and implementation specification only; NOT implemented, trained or experimentally verified**.

## 0. Authoritative baseline and identity

- Paper: `JLU-MCNS-MEC/2027-Qiaoxu-CVPR` main, `sec/4_method.tex`, `appendix/C_visual_evidence.tex`, `appendix/D_training.tex`.
- Code: this repository's main, `graphnav/full_localization.py`, `graphnav/stage1.py`, `graphnav/stage1_runtime.py`, `scripts/train_graphnav_stage1.py`, `scripts/evaluate_closedloop.py`.
- Original **formal Stage1**: map geometry + full instruction tokens -> A3 28×28 `B0`; frozen SigLIP2 384×384 observation -> 32/64/128m ROI vectors -> separate appearance/current/history branch -> bounded visual logit residual `Delta` -> `Bt`; Top-1 guarded-blend trajectory/Stop.
- The paper's separate CVR head is **not** part of the formal checkpoint. Do not add CVR here, and do not mix the separate local-box WALDO refinement experiment into this ablation.
- Original archived formal checkpoint 00f2acc0... and original results are **reference**, not a matched end-to-end-retraining control.

## 1. Precise hypothesis

RGB appeared weak in the independently applied residual because it was fused too late. Test **feature-level integration inside the language-conditioned map-geometric goal localizer** (before candidate–landmark attention, geometry scoring and Top-K), learning RGB, geometry and language together, rather than adding a separate visual correction to logits.

Only the RGB fusion architecture and the explicitly requested full-retraining regime may change. All CityNav inputs, mappings, original detector-free candidate definition, 28×28 output, known-candidate masks, causal past-history protocol, pretrained trajectory/Stop architecture, waypoint constraints, stopping threshold and evaluation splits are fixed.

## 2. New architecture (NOT a local-box reranker)

- Keep exact `FullCandidateDataset` inputs, candidate–landmark 10-D pair geometry, map priors, BERT token/name embeddings, null landmark anchor, Q=784 grid, K/top-K, 128-D candidate context, controller.
- Keep `ObservedPatchScaleEncoder` with current observed **384×384 orthophoto**, frozen SigLIP2 24×24 contextual tokens (D=1024), shared ROI pooling for 32/64/128m, validity and quality. Do not use GT/future/city-wide source raster.
- Preserve exactly the same past-only `CandidateMemory` and <2m duplicate-view rule. At step t use history features from steps strictly before t, then commit current features **after** the policy decision.
- Fuse current and historical supported candidate RGB vectors, using the original quality/validity evidence and a trainable gate within the **localizer**. Invalid ROI vectors are unknown, never negative observations.
- For every goal candidate q, project its supported RGB evidence to H=128 and inject the gated RGB representation into the **query state before token cross-attention and candidate–landmark relation scoring**; optionally use a language-conditioned candidate RGB interaction *within the same localizer*. Run geometry/text/RGB feature fusion jointly before predicting the candidate logit and dense 128-D context. This must affect both belief ranking **and** candidate context, not merely `logits += visual_delta`.
- **Delete/bypass the separate matcher-to-logit residual in this arm**. Do not apply original `Delta_F`, standalone CVR, a second correction head, or a parallel post-hoc reranker.
- Keep a separate map-only `B0` for diagnosis and zero-vision behavior. When no current/history ROI is valid, enforce **exact** fallback of the fused belief and candidate context to geometry-only output (including proposal ordering/ties). Compute B0 from public map+text only; no current or future RGB.
- RGB can only affect candidates with valid observed support via learned feature integration; keep unsupported candidate's *direct* features/scores identical to map-only values. Softmax normalization may change posterior probabilities globally.

Sketch (informative, not code):
```
Gq = geometry_and_landmark_query(q, map, instruction)
Vq, Mq = causal_quality_fuse(current_RGB_ROIs, past_RGB_ROIs)
Hq = language_landmark_fusion(Gq + Mq * Project(Vq), instruction, referenced_landmarks)
logit_q, context_q = same_joint_goal_localizer(Hq)
Bt = softmax(logits_0 ... logits_783)  # NO separate RGB residual
```

## 3. Training contract

- Do NOT fine-tune frozen BERT/SigLIP2: "full retraining" means all **trainable goal-localizer, new fusion/quality modules and trajectory/Stop parameters** are optimized jointly, not reopening pretrained foundation encoders.
- Start both matched arms from the **same verified** A3 Seed1 Epoch14 and original trajectory/Stop weights; initialize new RGB-fusion layers randomly with a fixed seed. Do NOT initialize the new model from the final 00f2acc0 checkpoint.
- Full/restart training, not a frozen-base one- or two-epoch adapter. No oracle target inputs, GT-boxes or additional WALDO supervision.
- Paper's A3/B3/C5 staged optimization has early frozen phases. To test the user's requested **joint full retraining from the outset**, train **both matched arms** (original separate-residual vs feature-level joint-fusion) with A3 + their RGB components + Trajectory/Stop trainable over the same 11-epoch budget; use matched seed(s), effective batch, LR schedule, optimization steps, observations and selected losses. Log any intentional loss modification explicitly, ideally keeping all meaningful goal/belief/visual-ranking/trajectory/Stop weights and supervision identical.
- Auxiliary visual ranking losses must not silently become a different task: if the original matcher-specific loss cannot be applied identically to a branch-free architecture, define an equivalent supervised candidate-discrimination auxiliary signal for BOTH matched arms and report the difference from the archived formal objective.
- Keep initial `L_static` geometry-only (B0) and `L_belief` on fused Bt with the same ground-truth goal location; preserve original 20m legal target, waypoint/Stop teacher labels, stop pos weight 20, `guarded-blend` and 0.3 stop threshold.
- Use the existing complete ~87,512 expert-prefix state cache if hash/protocol checks pass (original ~83,140 train / ~4,372 internal dev), respecting the original query-group split. No hidden Unseen selection. Recompute trainable A3 on each step; **do not reuse a `base_field` that blocks A3 gradients**.
- No overwriting original runs or main. New experiment output root `runs/joint_rgb_geometry_20261027`, immutable config/code/cache/checkpoint hashes, resume-safe status.

## 4. Required evaluation arms

| Arm | Initialization/training | RGB mechanism |
| --- | --- | --- |
| P0 archive | Frozen archived formal checkpoint, report only | original separate residual; context/reference |
| A0 paired-original | Verified original A3 + controller, 11-epoch joint full retrain | original residual |
| A1 **paired-joint** | Exactly matched A0 initialization, steps and training | RGB integrated in full candidate–landmark localizer, no post-hoc residual |
| A1-zero | Fixed A1 weights; inference-only control | zero current+history RGB features, preserve ROI validity/quality |
| A1-shuffle | Fixed A1 weights; inference-only control | shuffle supported ROI feature–candidate association within scale, preserve masks and pose |
| A1-current-only | Fixed A1 weights; inference-only control | no history, same current image |
| A2 geometry-only (optional) | Matched retrain | no RGB |

If budget is tight, prioritize A0 and A1 paired runs, one seed first; then multi-seed **without changing hyperparameters after Unseen**. A0 vs A1 is the causal comparison. Do not call P0 vs A1 a strict architecture-only ablation because the training schedule differs.

## 5. Hard acceptance checks

1. Exact original static B0 no-RGB parity, and exact fused==static fallback on an all-invalid input (including Top-K tie order and context); zero-RGB must be deterministic.
2. Perturb *supported* RGB and confirm nonzero gradients and changes in joint localizer features/logits; same perturbation on unsupported vectors has exactly zero effect. Perturb map geometry/text and confirm their gradients. Check unobserved never acts as negative.
3. Verify feature integration happens **before candidate–landmark attention / geometry scoring / Top-K**. Merely changing post-hoc `visual_gain` does NOT satisfy the experiment.
4. Verify source checkpoint SHA, row/annotation/encoder SHA, query split, history strictly past, no target ID, target coords, future image/pose in model input.
5. Ensure stage1 runtime loads new checkpoint by an explicit **new protocol/model name** rather than silently accepting it as `graphnav-v1-stage1`; the evaluation runner must use exactly one inference call to the original controller after computing the fused goal field.
6. Evaluate identical frozen episode manifests: Seen128 for checkpoint selection, then full Val-Seen 2470 and Val-Unseen 2697. Log SR, OSR, SPL, NE, nDTW, sDTW, false-stop, static/fused R@1/4/8, per-map paired bootstrap CI, changed Top-1/goal cells, fraction with valid RGB, wall-clock and VRAM.
7. Run CPU synthetic regression/smoke and at least one real CUDA forward/backward, then full budget; no performance claim before completed paired navigation.
8. Keep failures, intervention results and any parameter-count difference (joint arm may have higher capacity). For mechanism proof include **size-matched** A0 or capacity sensitivity follow-up if parameter count varies materially.

## 6. Expected code touchpoints / safe implementation sequence

1. Add `graphnav/joint_rgb_geometry.py` or a strictly opt-in extension of `FullTokenGeometryLocalizer`; preserve `geometry` and the original forward API defaults.
2. Add separate `JointRGBGeometryStage1` model wrapper and checkpoint identity; no change to original archive loading behavior.
3. Extend `scripts/train_graphnav_stage1.py` via new entrypoint or experiment-specific runner for paired 11-epoch joint optimization; preserve exact current/history data extraction.
4. Add `JointRGBGeometryPolicy` + explicit evaluation model argument to the shared `scripts/evaluate_closedloop.py`; do not change original `graphnav-v1` execution path.
5. Write and run tests before committing a "ready-to-train" label, then run full GPU training on the data-host machine. This branch currently contains the protocol only.

This experiment is **city-scale grid-candidate RGB fusion**, not instance-level box detection, 3D reconstruction, 384×384 goal-grid retraining, or a new controller.
