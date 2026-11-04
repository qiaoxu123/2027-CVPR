#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.evaluate_pgvb_local --data /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/data --training /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/pilot_s0 --old-training /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_structured_v1/pilot_s0 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/evaluation > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/evaluate.log 2>&1
printf '0\n' > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/evaluate.exit
