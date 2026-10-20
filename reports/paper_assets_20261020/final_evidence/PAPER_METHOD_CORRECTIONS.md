# Required manuscript corrections and implementation clarifications

Compared against paper commit `6e7450c7580a61e1158e7cfc5eb5ad66eea05905`, especially `sec/3_problem.tex` and `sec/4_method.tex`; hashes are in `MANUSCRIPT_SNAPSHOT.json`. The manuscript repository was not edited. All implementation details below refer to experiment commit `16452896e8b169a9c540ce6de7f7d00b17b453a7`, whose relevant files match archived inference hashes. Full functions, shapes and source anchors are in `MODEL_MECHANISM_AUDIT.md`.

## Required wording fixes

|Location|Current issue|Required replacement / qualification|
|---|---|---|
|§3, line 4; §4, line 31; any new RGB caption|Generic RGB/history description leaves the image acquisition model unspecified.|State that the evaluated RGB is a **pose-conditioned nadir orthophoto crop**, 384×384, with height-derived support; it is not onboard perspective FPV. Do not imply real optical occlusion or readable facade evidence.|
|§4, line 46|“map/elevation context” can imply local terrain/depth supplied to the controller.|Specify public XY bounds, a map-global ground-level scalar and current XYZ/yaw. The controller receives static 128-D candidate context, updated belief scores/order/entropy, and candidate-relative pose features; `observed_fraction=0`. It does not consume the local DEM or raw RGB.|
|§4, line 40; §5 visual-table caption line 102|“confidence-calibrated” implies confidence reliability beyond the available evidence.|Use **confidence-gated residual head** when describing the mechanism. Retaining the proper model name CVR is possible, but say its sigmoid gate has not been validated by ECE/reliability tests and is not an estimated semantic-visibility probability.|
|§4, lines 67–76 / Eq. training|Four-term navigation loss is not the complete recorded staged objective.|Explicitly label it a controller subobjective and provide the actual static/belief/visual/hard-rank and anchor losses below, stage freezing and independent CVR training. Do not imply one common jointly trained objective for all reported variants.|

These are precision/omission corrections. The current paper already labels several equations as abstractions; those abstractions should be grounded with the following operational definitions rather than described as nonexistent code.

## Add exact visual update definitions

After §3 Eq. concept_visual / §4 visual subsection, add an implementation paragraph such as:

> Candidate-to-image correspondence is provided by a world-XY to observed-frame homography using the current pose-conditioned footprint. Three 32/64/128-m candidate squares pool contextual patch tokens from one frozen full-frame SigLIP2 encoding. Valid evidence requires in-frame ROI geometry, sufficient projected size and nonblack/depth-valid pixel and patch support. It does not certify target-instance visibility. Historical state comprises per-candidate quality-weighted feature averages, read before the current observation is inserted.

For the **formal** model, with current/history support union \(m_t^F\), quality-fused matcher score \(a_t\) and supported mean \(\bar a_t\), the actual equations are

\[
\Delta_t^F(q)=0.5\tanh(g_v)\tanh((a_t(q)-\bar a_t)/5)m_t^F(q),\quad
B_t=\operatorname{softmax}(s_0+\Delta_t^F).
\]

For the **independent CVR**, the mask additionally requires a nonzero current/history visual feature. Its 106-dimensional input includes static context, image–text products, current/history ROI quality, static scores/probability, matcher scores, support and UAV distance. The head predicts raw residual r and confidence-gate logit z:

\[
\Delta_t^C(q)=2\tanh(r_t(q)-\bar r_t)\sigma(z_t(q))m_t^C(q),\quad
B_t^C=\operatorname{softmax}(s_0+\Delta_t^C).
\]

CVR **replaces**, rather than stacks on top of, the formal residual. Its observed-mask export is still the formal mask; the extra nonzero-feature gate must not be inferred from that export. Neither model recursively adds a new residual to previous belief logits. The fixed static field is reused; feature memory, not the goal distribution, is the recurrent state. Unknown cells have zero direct residual but can change probability through normalization.

## Add actual staged training description

With losses named as in `graphnav/stage1.py:128–151`:

\[
L_{A/B}=L_{static}+L_{belief}+0.1L_{visual}+0.05L_{hardrank},
\]
\[
L_C=L_{A/B}+0.1L_{path}+0.02L_{mode}+0.05L_{navrank}+0.05L_{stop}+L_{anchor}.
\]

Stages A/B freeze the original static localizer and controller; history is enabled/trainable in B/C, while appearance matching is trainable throughout. Stage C unfreezes active static/controller branches. Inactive GAT/old-vision branches remain frozen. BERT and SigLIP2 remain frozen. Recorded schedule A3/B3/C5 selects C epoch1. The independent three-epoch CVR freezes the original modules and uses 20-m goal-mass NLL +0.1 non-improvement penalty +0.02 supported residual drift; default rank-loss weight is zero. Do not promote later optional CLI contrast/ranking flags to selected-model training facts. Train/dev states and config provenance are recorded in the mechanism audit.

## Controls and controller details to make explicit

- **Zero** sets current and historical feature vectors to zero while retaining ROI geometry/quality. In CVR this also closes the nonzero-feature gate, so it is not a content-only intervention with identical effective support.
- **Shuffle** permutes supported vectors across candidates per scale, keeping receiver geometry, quality and pose. It probes spatial association and content jointly, not only semantic understanding.
- **No-belief** removes the residual, not all encoder/matcher/history computation.
- The deployed trajectory/Stop head is recomputed from the **updated** belief. The forward-returned `output.trajectory` is stale after the CVR wrapper and is not the executed command. Top-1 belief selects the goal in guarded-blend; the trained candidate-ranking head does not override it in this branch.
- Dense NumPy argmax, retrieval `torch.topk`, and preserved controller proposal tie ordering can differ for equal logits. Label plots as dense Top-1 when appropriate; use saved selected goals for execution claims. A diagnostic Top-16 export does not prove the controller's checkpoint K.

## Current claims/captions that are already consistent

The paper already separates the formal main checkpoint, independent CVR, dense candidate grid and separately trained geometry heads (§4 line5, §5 line8). Its Eq. concept_visual uses a fixed s0 and correctly notes global normalization; it does not claim recursive Bayes. The 384×384 intervention is correctly called a candidate grid, not RGB resolution (§4 line27). The Stop equation, 0.001-m predicted-goal guard, 20-m evaluation distinction, one-waypoint execution and ≤25% guarded blend are already correct. Do not “correct” them into a different policy.

`figures/overview.tex` already states that success alone does not establish visual gain; `figures/visual_evidence.tex` correctly labels independent CVR and matched-path controls and warns that Shuffle can retain Top-1. The geometry caption correctly distinguishes separately trained arms and source coordinates from geographic north. The navigation caption correctly avoids equating OSR=1/SR=0 with a causal Stop failure. The new evidence supports these qualifications.

## Unsupported extensions to exclude

Do not claim perspective/occlusion-aware visibility, raw-RGB temporal attention, semantic correspondence independent of known pose, formal-mode GAT message passing, active information-gain planning, recursively accumulated goal probabilities, learned confidence calibration, or proven semantic visual understanding. None is established by the inspected policy and saved evidence. New strong semantic claims require a support/pose/quality-preserving content intervention or other independent semantic evidence; this task did not generate such experiments.
