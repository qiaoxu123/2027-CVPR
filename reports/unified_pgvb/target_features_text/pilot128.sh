#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
BASE=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs
RUN=$BASE/unified_v2_target_features_s42_20261106
SBF=/home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source
"$PY" -u -m scripts.evaluate_closedloop --sbf-source "$SBF" --config "$SBF/configs/sbfnav/main.yaml" --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --split val_seen --max-episodes 128 --diagnostic-pilot --model unified-relation --checkpoint "$BASE/graphnav_v1_stage1/best.pt" --unified-v2 "$BASE/unified_pgvb_v2_pilot_s42/full_training/last.pt" --relation-checkpoint "$RUN/train/last.pt" --relation-control full --relation-visual-control off --instruction-mode full_text --public-objects /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --output "$RUN/pilot128" > "$RUN/pilot128.log" 2>&1
