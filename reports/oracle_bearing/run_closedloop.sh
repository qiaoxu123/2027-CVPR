#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-oracle-bearing
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
RUN=$BASE/oracle_bearing_20261106
SBF=/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
DATA=/home/20261024_1/Workspace/DATA/sbfnav-data
for ARM in "$@"; do
 RESUME=()
 if [ -f "$RUN/nav_$ARM/full_relation_manifest.json" ]; then RESUME=(--resume); fi
 "$PY" -u -m scripts.run_oracle_bearing_closedloop --oracle-arm "$ARM" --oracle-angles "$RUN/angles_ONLY_ORACLE.json" --model unified-relation --split val_seen --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" --data-root "$DATA" --checkpoint "$BASE/graphnav_v1_stage1/best.pt" --unified-v2 "$BASE/unified_pgvb_v2_pilot_s42/full_training/last.pt" --relation-checkpoint "$BASE/unified_v2_target_features_s42_20261106/train/last.pt" --instruction-mode full_text --public-objects "$DATA/cityrefer_original/objects.json" --relation-control full --relation-visual-control off --relation-geometry-source trained --evaluation-concurrency 2 --output "$RUN/nav_$ARM" "${RESUME[@]}" > "$RUN/nav_$ARM.log" 2>&1
done
