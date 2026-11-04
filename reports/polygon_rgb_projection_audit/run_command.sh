#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.audit_polygon_rgb_projection --source-data /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/geometry_guided_visual_belief/landmark_data_augmented_v1 --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_projection_audit_20261104/output_v2 > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_projection_audit_20261104/run_v2.log 2>&1
printf '0\n' > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_projection_audit_20261104/run.exit
