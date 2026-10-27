## [LocalSemanticRelationRefinement 0.1] - 2026-10-27
### Features
- Independent language-conditioned local box/reference relation network, observed RGB-D feature cache, masked instance supervision, training/calibration and original28×28 policy adapter.
- Local probability mass preservation, semantic/distance probes, zero-branch controller parity and reproducible remote bundle.
### Design Rationale
- Keep original macro localization and flight controller fixed; use full frozen language/RGB features and explicit relations for local instance refinement.
### Notes & Caveats
- DSM-derived height is weak observed-surface evidence; strict instance matching yields limited supervision. Implementation verification is not navigation efficacy evidence. Unqualified/diagnostic gates remain disabled.

## [BoxRelationCorrection 0.1] - 2026-10-27
### Features
- Separately trained semantic/bearing/observed-surface-height box correction branch and optional original GraphNav-V1 policy integration, using the user-selected paper-native 28×28 configuration.
- Fixed Seen/Unseen paired128 baseline/correction navigation plus Seen semantic-only/no-angle/no-height controls; raw rollouts and measured heatmap changes retained.
### Design Rationale
- Freeze the paper checkpoint and controller; insert a bounded, observed-support correction before the single original controller call. Explicit relation supervision and masked height observations permit independent diagnosis.
### Notes & Caveats
- Paired128 SR changes: Seen 50.00→51.56%, Unseen 20.31→22.66%; small discordant counts, Seen SPL slightly lower, height ablation shows no SR gain, and added perception increases latency. No full-paper improvement claim or default checkpoint replacement.
- Mandatory adversarial-review invocation failed with workspace routing discovery timeout; supplemental audit fixes and execution tests retained in Experiments/BoxRelationCorrection.

## [CropSemanticResidualE 0.1] - 2026-10-26
### Features
- Independent crop semantic pretraining and E0/E1 bounded visual residual experiment.
### Design Rationale
- Train target-crop/text correspondence and bridge to existing ROI features before joint navigation; preserve original architecture and controller.
### Notes & Caveats
- Single seed and limited budget; no benefit assumed. Original experiments and paper unchanged. Execution status and all outcomes live in Experiments/CropSemanticResidualE.

## [BeliefROISemantics 0.1] - 2026-10-26
### Features
- Independent belief-derived 20m neighborhood type recognition with paired original384/native-detail crops and frozen-main-model navigation integration.
### Design Rationale
- Test real image sampling and region semantics before navigation; keep original geometry, visual matcher, history and controller frozen to avoid joint-training drift. No diffusion or GT-chosen crops.
### Notes & Caveats
- Detail observations change image sampling permissions; annotation-derived negatives are not guaranteed exhaustive. Sparse exact-center corrections have limited negative-rejection coverage. Current execution status and all positive/negative outcomes live in Experiments/BeliefROISemantics; original paper and weights unchanged.

## [StrongVLM ROI smoke 0.1] - 2026-10-26
### Features
- Isolated pinned Qwen3-VL-32B deployment and fixed-weight, tri-state ROI type exclusion smoke tests with original/detail crops, blank controls and correct-target retention cases.
### Design Rationale
- Test a stronger open foundation model before additional training; keep candidate generation, original navigation weights and paper unchanged. Public image-only inference is separated from offline labels.
### Notes & Caveats
- Small stratified cases and retrospective regression are not navigation gains; Ground checks include training-source images. Execution status, costs and complete outputs live in Experiments/BeliefROISemantics/StrongVLM/reports.

## [High-detail ROI Navigation 0.1] - 2026-10-26
### Features
- Frozen original-model versus high-detail ROI region-correction navigation comparison on the original Seen128/Unseen128 manifests.
### Design Rationale
- Measure actual SR/SPL and paired help/harm rather than infer navigation value from ROI recognition. Use four-view consensus, explicit support checks, and bounded region penalties with the original controller recomputed from the updated field.
### Notes & Caveats
- Extra detail observations and regional correction are a combined system change. No retraining or paper edits. Strict-support prototypes and partial runs are archived separately; all outcomes are retained in Experiments/BeliefROISemantics/NavigationV2.

## [BeliefROIMask probe 0.1] - 2026-10-26
### Features
- Independent real-frame 384/768/1152 SigLIP2 spatial-token probe and belief-region mask visualization in Experiments/BeliefROIMask.
### Design Rationale
- Verify shared high-resolution encoding and existing world/frame projection before training a correction readout or integrating navigation.
### Notes & Caveats
- One training-frame feasibility check, no recognition/navigation improvement claim. Position interpolation explicit; first-pass timings are not comparable steady-state benchmarks. Original weights, active evaluation code and paper unchanged.

## [BeliefROIMask adaptation 0.2] - 2026-10-26
### Features
- Isolated 768px mask-feature extraction and paired 4100-parameter crop/mask readout training, frozen vision backbone and equal 128x16 update budget.
### Design Rationale
- Compare domain-adapted readouts on the same supported training candidates; select checkpoints and rejection thresholds only on internal selection data.
### Notes & Caveats
- Existing retrospective holdout: crop rejects16/35 wrong-type candidates, mask8/35; both target harm0/14. No navigation improvement claim. Unsupported regions abstain; original weights and paper preserved.

## [Single high-detail crop navigation 0.1] - 2026-10-26
### Features
- Independent current-frame single-crop type correction with frozen adapted head, bounded center-only penalties and actual original-controller recomputation.
- Full Seen2470/Unseen2697 evaluation coordinator with internal speed gate and original-result pairing; independent color-label feasibility audit.
### Design Rationale
- Reuse actual paper checkpoint's28×28 goal grid and verify old trajectory parity; preserve384 observation input. Earlier ROI experiments'384 goal grid must not be directly compared to paper28 results.
### Notes & Caveats
- Full evaluation started, not completed; no navigation gain claimed. Center-only penalties may be bypassed by nearby unchecked goals. Color remains offline-only; no reliable independent color labels. Original results and paper unchanged.

## [Single crop navigation equivalent speed fix 0.1.1] - 2026-10-26
### Features
- NavigationSingleFast shares current-frame pixel-validity preparation across candidate crops; complete-split evaluation restarts in isolated output.
### Design Rationale
- Remove repeated full-frame scans without changing crop pixels, ROI support, model weights or thresholds. Internal rollout mean wall/decision reduced0.874→0.357s.
### Notes & Caveats
-39 real candidate crops and two internal trajectories/probabilities exactly match. Original slow partial results preserved; no Val-performance-based policy changes. Complete navigation metrics pending.

## [Color recognition exploratory probe 0.1] - 2026-10-26
### Features
- Train-internal weak car-color joint head, frozen SigLIP text baseline and fixed-prompt Qwen3-VL-32B image-only comparison; original type head unchanged.
### Design Rationale
- Separate type presence from type-color conjunction; ignore unknown labels, isolate train/selection/heldout spatial support, calibrate only selection.
### Notes & Caveats
- Six complete weak-label ROIs from three episodes are insufficient for a validated accuracy claim. Small head Top1 collapses to white; Qwen promising on examples but ~3.4s/ROI. No color navigation integration or Unseen restart; shape/material untested.

## [Small VL color pilot 0.1] - 2026-10-26
### Features
- Pinned Qwen3-VL-2B deployment, identical-ROI baseline, compact-output baseline and isolated rank8 language LoRA; original navigation unchanged.
- Teacher/text-agreement weak supervision, train-only38 retained ROIs and internal9 selection, exploratory6 heldout.
### Design Rationale
- Measure both speed and harmful rejection; isolate shorter output format from model-size effects. Preserve frozen vision/base weights, small adapter only.
### Notes & Caveats
- LoRA improved format validity but increased heldout color harm; no navigation integration or Unseen restart. EarthDial4B domain-specific comparison prepared in isolated dependencies; weights download/test status recorded separately.

## [Small VL single-query comparison 0.1.1] - 2026-10-26
### Features
- Completed author-code EarthDial4B RGB comparison in isolated Transformers4.40 dependencies and matched single-color yes/no tests against Qwen2B before/after LoRA.
### Design Rationale
- Navigation asks about a specific attribute; separately test this interface instead of assuming long JSON or compact8-symbol behavior transfers.
### Notes & Caveats
- Exploratory same6 weak-label ROIs: EarthDial rejects6/40, Qwen2B16/40, adapted2B18/40; all harm0/8 in this interface. Approx0.06s per2B query excludes preprocessing/navigation. Format sensitivity and weak tiny evaluation preclude validated improvement claim or automatic navigation integration.

## [Small VL navigation pilot 0.1] - 2026-10-26
### Features
- Isolated NavigationSmallVL plugs frozen Qwen3-VL-2B LoRA into live observed belief ROI correction.
### Design Rationale
- Explicit color-car instructions only; no response lowers the checked center by fixed 0.5; yes/unknown abstain. Original controller recomputes from corrected belief. Paired first eight internal development episodes; no Unseen restart.
### Notes & Caveats
- Exploratory development check, not heldout generalization. No claim of calibrated negatives; original weights and paper unchanged.
