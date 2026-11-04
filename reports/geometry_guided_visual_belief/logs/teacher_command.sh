#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.cache_landmark_geometry_teacher \
 --data /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/geometry_guided_visual_belief/landmark_data_augmented_v1 \
 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data \
 --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source \
 --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt \
 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/geometry_guided_visual_belief/teacher_v1 \
 > /tmp/geometry_guided_visual_belief/teacher.log 2>&1
printf '0\n' > /tmp/geometry_guided_visual_belief/teacher.exit
