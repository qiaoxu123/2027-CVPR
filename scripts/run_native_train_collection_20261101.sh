#!/usr/bin/env bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.collect_native_onpolicy \
 --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source \
 --config /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml \
 --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt \
 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --split train_seen --model graphnav-v1 \
 --train-collection-manifest aerialvg_objects_train256_20261101/manifest.json \
 --native-checkpoint aerialvg_native_plain_sigma8_s0_20261101/best.pt \
 --max-episodes 128 --output aerialvg_native_onpolicy_train128_20261101
