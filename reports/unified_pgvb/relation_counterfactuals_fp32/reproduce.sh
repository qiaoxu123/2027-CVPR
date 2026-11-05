#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.run_relation_counterfactuals --source /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --official-source /home/20261024_1/Workspace/third_party/citynav_protocol_reference --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_counterfactuals_fp32_20261105/results_final > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_counterfactuals_fp32_20261105/run_final.log 2>&1
