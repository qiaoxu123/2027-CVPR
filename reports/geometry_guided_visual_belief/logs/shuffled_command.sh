#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.run_shuffled_landmark_regression \
 --selection /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/named_landmark_navigation/selection64_v1 \
 --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt \
 --visual-checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/geometry_guided_visual_belief/pilot_s0_v1/final.pt \
 --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source \
 --config /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml \
 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data \
 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/geometry_guided_visual_belief/shuffled_landmark_regression64_v1 \
 > /tmp/geometry_guided_visual_belief/shuffled_regression.log 2>&1
printf '0\n' > /tmp/geometry_guided_visual_belief/shuffled_regression.exit
