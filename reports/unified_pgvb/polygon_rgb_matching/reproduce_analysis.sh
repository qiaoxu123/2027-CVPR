#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -m scripts.analyze_polygon_rgb_matching --run /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/polygon_rgb_matching_20261105/results --nearfield reports/unified_pgvb/nearfield_scene_audit --rollout /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/stable_landmark_20261105/full64 --output reports/unified_pgvb/polygon_rgb_matching
