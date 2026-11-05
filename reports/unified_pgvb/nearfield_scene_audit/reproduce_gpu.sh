#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -m scripts.build_nearfield_scene_dataset --run /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/stable_landmark_20261105/full64 --selection /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/named_landmark_navigation/selection64_v1 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --stage1 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/nearfield_scene_20261105/dataset_reproduction --resume
