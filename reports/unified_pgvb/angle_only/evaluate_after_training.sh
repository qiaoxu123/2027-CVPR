#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
RUN=$BASE/unified_angle_only_s42_20261106
SBF=/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data
while [ ! -f "$RUN/full/report.json" ]; do
 if ! tmux has-session -t unified_angle_training 2>/dev/null; then echo 'Training ended without completion report';exit 1;fi
 sleep 10
done
"$PY" -u -m scripts.smoke_unified_angle_runtime --stage1 "$BASE/graphnav_v1_stage1/best.pt" --original-v2 "$BASE/unified_pgvb_v2_pilot_s42/full_training/last.pt" --checkpoint "$BASE/unified_full_relation_s42_20261105/core/last.pt" --angle-checkpoint "$RUN/full/last.pt" --public-objects "$DATA/cityrefer_original/objects.json" --data-root "$DATA" --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" --output "$RUN/runtime_smoke" > "$RUN/runtime_smoke.log" 2>&1
"$PY" -u -m scripts.freeze_angle_only --run "$RUN" > "$RUN/freeze.log" 2>&1
"$PY" -u -m scripts.evaluate_closedloop --model unified-relation --split val_seen --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" --data-root "$DATA" --checkpoint "$BASE/graphnav_v1_stage1/best.pt" --unified-v2 "$BASE/unified_pgvb_v2_pilot_s42/full_training/last.pt" --relation-checkpoint "$BASE/unified_full_relation_s42_20261105/core/last.pt" --relation-control full --relation-visual-control off --relation-geometry-source trained --angle-head-checkpoint "$RUN/full/last.pt" --instruction-mode full_text --public-objects "$DATA/cityrefer_original/objects.json" --output "$RUN/val_seen_predicted_angles" > "$RUN/navigation.log" 2>&1
