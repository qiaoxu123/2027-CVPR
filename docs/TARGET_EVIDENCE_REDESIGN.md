# GraphNav target evidence redesign

Status: opt-in experimental implementation. No new SR or SPL result has been measured. The parent branch is 2027-CVPR/graphnav-closedloop-sbf-parity; its checkpoints and A3 behavior are unchanged.

## Implemented

- graphnav/evidence_grounding.py: candidate-wide relation-residual scoring; appearance-query extraction; BERT-token to observed SigLIP2 patch matching with trainable cross-modal projections; patch-to-public-map sparse rasterization; causal spatial evidence memory with pose deduplication; zero-initialized gates preserving A3.
- graphnav/target_runtime.py: opt-in TargetEvidenceWaypointPolicy subclass of DynamicWaypointPolicy. It uses only current observed RGB, source-valid depth, public camera footprint and public pose. It updates 28x28 goal belief before Top-K. A separate trained checkpoint is mandatory.
- graphnav/train_target_evidence.py: isolated Train Seen CE plus A3-preserving KL objective; per-checkpoint source SHA checks; private target labels accepted only inside the loss.
- tests/test_evidence_grounding.py, tests/test_target_runtime.py: masking, unchanged A3 at zero gates, memory, train-only splits, observed camera projection.

## Required training data

The adapter is NOT pretrained. Generate an audited Train Seen torch file with split=train_seen, causal_observations=true, and a list of rows. Each row holds tensors with batch dimension one:

base_logits [1,Q], dense_context [1,Q,H], grid_xy [1,Q,2],
relation_fields [1,Q,R], text_tokens [1,L,768], text_mask [1,L],
patch_features [1,P,D], patch_xy [1,P,2], patch_valid [1,P],
target_distribution [1,Q].

Generate visual features from ACTUALLY visited Train RGB views, never target omniscient imagery. Keep source image path and observation cutoff provenance in a sidecar audit. The training script checks tensor/split fields, but an independent preparer must check actual observation provenance. Add self-rollout prefixes to address expert-only training distribution mismatch.

Example after preparing audited rows:

    python -m graphnav.train_target_evidence --rows data/evidence_train_seen.pt --a3-checkpoint runs/a3/best.pt --output runs/target_evidence/best.pt --epochs 5 --device cuda

To enable at runtime, instantiate graphnav.target_runtime.TargetEvidenceWaypointPolicy with the same existing DynamicWaypointPolicy arguments and the additional evidence_checkpoint argument. The old CLI is unmodified and does not auto-enable this policy.

## Honest limitations

- relation_fields are inherited from the current A3 pipeline; full compositional ordinal/left/right parser is not implemented here.
- The new target query uses BERT appearance tokens, NOT native pretrained SigLIP2 text embeddings, so the trainable cross-modal projection requires real supervision.
- Source observations use orthophoto plus vertical-height depth proxy; no perspective3D/occlusion/swept safety claim.
- Pixel-to-grid evidence is sparse, view-constrained and 28x28 coarse; it is not actual semantic segmentation.
- History is pose-deduplicated grid evidence, not recurrent graph/pose attention.
- There is no new trained adapter checkpoint or validated performance result.

## Validation gates

Run pytest on this branch first. Compare exact A3 fallback, relation-only, appearance-only, joint, real RGB, zero visual and shuffled visual under the SAME Seen Episode manifest. Report target rank shifts, map-wise bootstrap, 2470 Seen closed-loop SR/SPL/OSR/NE, false stops, and runtime. Freeze selection without inspecting GraphNav Unseen, then test 2697 Unseen raw episodes. Do not claim an improvement from frozen encoder retrieval accuracy alone.


## New planned phase: multi-scale and active observation

See [ACTIVE_MULTISCALE_PERCEPTION_PLAN.md](ACTIVE_MULTISCALE_PERCEPTION_PLAN.md) for
the *pre-registered* evaluation ladder: raw-image/source-GSD audit; frozen
SigLIP2 native-text target-appearance retrieval; source-faithful observed ROI
scales; matched fixed descent/approach actions; uncertainty-conditioned active
approach/descend/lateral controls; matched-budget full Seen/Unseen closed-loop.

The current branch adds `observed_scales.py` (no fake information from image
upsampling), `active_observation.py` (bounded goal-belief-only XY/Z candidate
actions), `--model target-evidence --active-observation` evaluation flags, and
related tests. Opt-in active actions use the *same* 20-step episode budget,
maximum 50m XY and 10m Z controller bounds. All novel gains remain unmeasured.
No native multiscale ROI *matching head* has yet been trained or connected.

**Camera caveat:** current nadir orthophoto cannot support claims about genuine
3D side views or additional scene occlusion cues from yaw/pitch. A separate
perspective camera simulator is required for physical viewpoint comparison.
