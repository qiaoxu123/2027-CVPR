#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.probe_pgvb_local_rgb_content --data /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/data --training /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/pilot_s0 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/content_probe > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/content_probe.log 2>&1
printf '0\n' > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pgvb_local_anchor_v2/content_probe.exit
