#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.prepare_pgvb_local_cache --source /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/geometry_guided_visual_belief/landmark_data_augmented_v1 --old-data /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_structured_v1/data --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json --backbone /home/20261024_1/.cache/huggingface/hub/models--google--siglip2-large-patch16-384/snapshots/1b426889ea62b5a72bf9839009a1b184bfc9c178 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/data > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/prepare.log 2>&1
