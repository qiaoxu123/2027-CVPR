#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -m scripts.run_polygon_rgb_matching --nearfield reports/unified_pgvb/nearfield_scene_audit --rollout /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/stable_landmark_20261105/full64 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --stage1 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_rgb_matching_20261105/results --resume
