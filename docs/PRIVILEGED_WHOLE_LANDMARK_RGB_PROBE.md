# Full Landmark RGB: frozen PGVB diagnostic (PRIVILEGED)

## Question

The UAV initially knows PUBLIC named landmark Polygon contours. **If an
image containing the whole landmark were also available at the start**, would
the already-trained PGVB head correct Goal Belief more reliably than the
actual initial local RGB? This is an upper-bound / oracle input diagnostic,
**not a legitimate causal navigation policy**: the CityNav environment's full
orthophoto is private to the renderer, and not part of the public map inputs
provided to the agent.

## Paired experimental design

For every original Val-Seen episode, at its exact starting pose, with one
unchanged pretrained checkpoint (full Relation + PGVB):

1. **Static:** Geometry+Relation, RGB disabled.
2. **Current:** one ACTUAL 384x384 camera RGB at the initial pose.
3. **Whole:** one PRIVILEGED 384x384 orthorectified RGB from the environment
   source orthophoto that covers the **whole public named Polygon** plus a
   25% context margin, with no GT target-aware crop selection.

The current and whole inputs use the SAME existing
`encode_landmark_observation`, frozen SigLIP2, PGVB and
`forward_full_relation`. **Both have the temporal LandmarkAcquisition gate
OFF** to isolate image content (otherwise a single full-map crop cannot pass
a 2-frame real-camera confirmation gate). Therefore this is **not a replay
of the original tracked-stable control policy**, and R@20m/NE are
**initial Goal estimation** statistics, NOT closed-loop SR/OSR/SPL.

- Unique supported Building/Road public reference: evaluate Current and Whole.
- Any nonunique/missing/unsupported reference: report why and preserve static
  predictions for both visual arms; keep the task in the ALL-task denominator.
- Full-image crop is made from public contour coordinates ONLY; private
  target coordinates and posthoc scores are never read before predictions.
- Full Polygon and surrounding source RGB are *privileged image input*;
  this cannot be presented as normal CityNav inference or as a valid
  comparison with causal-navigation benchmarks.
- Map height validity for this **orthophoto** is not perspective camera
  range/depth; depth_m is NaN and its synthetic origin is annotated.
- Public Polygon projection does not prove target-instance correctness.
- One huge polygon rectified into 384px may reduce object resolution; report
  nominal world meters per SigLIP2 patch. High-resolution tiling is a
  separate follow-up experiment, not something claimed in this run.

## Exact execution

Use the environment and Checkpoint paths from the recorded
`reports/unified_pgvb/full_relation_s42_20261105/final_evaluation.sh`.
Run from the same GraphNav repository root, first smoke then separate full
output directories:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105
STAGE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt
V2=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt
SBF=/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data

"$PY" -m pytest -q tests/test_privileged_landmark_rgb.py
"$PY" -u -m scripts.probe_full_landmark_rgb \
  --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" \
  --data-root "$DATA" --checkpoint "$STAGE" --unified-v2 "$V2" \
  --relation-checkpoint "$RUN/visual/last.pt" \
  --max-episodes 32 --output "$RUN/oracle_landmark_rgb_smoke"

"$PY" -u -m scripts.probe_full_landmark_rgb \
  --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" \
  --data-root "$DATA" --checkpoint "$STAGE" --unified-v2 "$V2" \
  --relation-checkpoint "$RUN/visual/last.pt" \
  --output "$RUN/oracle_landmark_rgb_full"
```

After interruption, rerun the exact command with `--resume`, preserving
the same `--output` and checkpoint. The frozen input/provenance manifest
must match.

Output:
- `manifest.json`: immutable source/weights/code fingerprint and explicit
  oracle-input label.
- `initial_goal_pairs.jsonl`: per-task 3-arm predictions, error, exclusion
  reason, whole landmark size and patch scale.
- `summary.json`: ALL task and uniquely supported subgroup R@20m/NE,
  corrections, damages, changed Top-1, camera/anchor eligibility counts.
- Optional `--save-paired-images N`: local real-camera vs privileged
  orthophoto PNGs. **Do not publish licensed source orthophoto crops.**

## Decision criteria (diagnostic only)

- If **Whole > Current** among the exact same supported cases, and the
  gain corresponds to coverage / sufficient valid patches, missing observations
  are a plausible bottleneck. Then test causal multi-frame accumulation or
  active public-Polygon-driven imaging, **without future/GT frame selection**.
- If **Whole ~= Current** despite good crop coverage, audit PGVB target identity
  conditioning, source-Patch effective resolution, Polygon binding and the
  learned visual score (including null effects). Full footprint alone does
  not prove instance recognition.
- If **Whole < Static** or produces more damages, it demonstrates that
  privileged visual coverage does not automatically repair this PGVB.
- Do not use Val scores to tune a new model or choose a checkpoint.
  For development/tuning, construct an analogous TRAIN-only diagnostic.

The current experiment never changes original Geometry, Relation, PGVB
weights, navigation controller, Stop/Goal thresholds, or main branches.
