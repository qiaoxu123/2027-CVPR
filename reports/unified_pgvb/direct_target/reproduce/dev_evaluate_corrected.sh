#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.evaluate_direct_target --inventory /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105/inventory_validated --cache /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105/cache_aligned --checkpoint /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105/training_aligned/best.pt --original-v2 /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105/dev_evaluation_corrected > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105/dev_evaluation_corrected.log 2>&1
