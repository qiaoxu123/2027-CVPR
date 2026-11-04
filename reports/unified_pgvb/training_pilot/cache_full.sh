#!/bin/bash
set -uo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.cache_unified_pgvb_train --annotations /home/20261024_1/Workspace/DATA/sbfnav-data/original_citynav/citynav_train_seen.json --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --old-cache /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/data --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_train_t0 > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/cache_full.log 2>&1
printf '%s\n' "$?" > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/cache_full.exit
