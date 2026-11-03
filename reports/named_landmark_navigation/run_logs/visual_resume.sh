#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.run_visual_only_landmark_navigation \
 --selection /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/named_landmark_navigation/selection64_v1 \
 --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt \
 --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source \
 --config /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml \
 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data \
 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/named_landmark_navigation/visual_only64_v1 \
 --resume > /tmp/named_landmark_navigation/visual_full.log 2>&1
printf '0\n' > /tmp/named_landmark_navigation/visual_full.exit
