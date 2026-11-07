#!/usr/bin/env bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-osm
exec /home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.audit_v2_candidate_observability --data /home/20261024_1/Workspace/DATA/sbfnav-data --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --initial-eval /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/osm_landmark_v2_recall_20261107/v2 --navigation /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/v2_public_navigation_20261107 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/v2_public_navigation_20261107/observability "$@"
