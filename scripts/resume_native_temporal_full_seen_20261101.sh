#!/usr/bin/env bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
native_full_python=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
"$native_full_python" -u -m scripts.evaluate_native_object_closedloop \
 --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source \
 --config /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml \
 --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt \
 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --split val_seen --model graphnav-v1 \
 --native-checkpoint aerialvg_native_plain_sigma8_s0_20261101/best.pt --native-history-window 8 \
 --native-frozen-manifest aerialvg_native_temporal_frozen_seen_candidate_20261101.json \
 --evaluation-concurrency 1 --resume-existing --output aerialvg_native_temporal_navigation_full_seen_20261101
"$native_full_python" -m scripts.summarize_native_navigation \
 --baseline /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/full_seen \
 --candidate aerialvg_native_temporal_navigation_full_seen_20261101 \
 --output aerialvg_native_temporal_full_seen_paired_20261101.json

"$native_full_python" -m scripts.verify_native_navigation_result --directory aerialvg_native_temporal_navigation_full_seen_20261101 --frozen aerialvg_native_temporal_frozen_seen_candidate_20261101.json --expected-episodes 2470 --require-full
