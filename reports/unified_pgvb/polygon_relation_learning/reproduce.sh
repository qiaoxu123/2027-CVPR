#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.train_polygon_relation --source /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_relation_learning_20261105/training > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_relation_learning_20261105/run.log 2>&1
