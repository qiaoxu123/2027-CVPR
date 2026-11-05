#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -m scripts.analyze_nearfield_scene --dataset /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/nearfield_scene_20261105/dataset_v2 --output reports/unified_pgvb/nearfield_scene_audit
