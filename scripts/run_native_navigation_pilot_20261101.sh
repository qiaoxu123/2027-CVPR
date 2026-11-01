#!/usr/bin/env bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
native_task_python=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
common=(--sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --config /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --split val_seen --model graphnav-v1 --stage1-diagnostic --stop-threshold .3 --head-execution guarded-blend --evaluation-concurrency 1)
"$native_task_python" -u -m scripts.evaluate_native_shared_closedloop "${common[@]}" --max-episodes 4 --output aerialvg_original_navigation_smoke4_20261101 > aerialvg_original_navigation_smoke4_20261101.log 2>&1
"$native_task_python" -u -m scripts.evaluate_native_shared_closedloop "${common[@]}" --max-episodes 128 --output aerialvg_original_navigation_seen128_20261101 > aerialvg_original_navigation_seen128_20261101.log 2>&1
"$native_task_python" -u -m scripts.evaluate_native_object_closedloop "${common[@]}" --native-checkpoint aerialvg_native_plain_sigma8_s0_20261101/best.pt --native-control real --max-episodes 128 --output aerialvg_native_navigation_seen128_20261101 > aerialvg_native_navigation_seen128_20261101.log 2>&1
