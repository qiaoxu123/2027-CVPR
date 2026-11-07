#!/usr/bin/env bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-osm
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.match_public_citynav_landmarks --data /home/20261024_1/Workspace/DATA/sbfnav-data --batch /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/osm_batch34_20261107 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/osm_landmark_v2_recall_20261107/landmarks

# Independent full V2 run (no OSM dependency):
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-osm
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.evaluate_v2_public_candidate_recall --source-v2 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --trained-v2 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105/core/last.pt --data /home/20261024_1/Workspace/DATA/sbfnav-data --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/osm_landmark_v2_recall_20261107/v2
