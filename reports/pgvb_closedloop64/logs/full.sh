#!/bin/bash
set -uo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.run_pgvb_landmark_closedloop \
 --selection /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/named_landmark_navigation/selection64_v1 \
 --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/graphnav_v1_stage1/best.pt \
 --pgvb-checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/pilot_s0/best.pt \
 --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source \
 --config /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml \
 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data \
 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_named_closedloop64_v1 \
 --resume --methods E4_PGVB E5_PGVB_only > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_closedloop64_execution_20261104/full.log 2>&1
printf '%s\n' "$?" > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_closedloop64_execution_20261104/full.exit
