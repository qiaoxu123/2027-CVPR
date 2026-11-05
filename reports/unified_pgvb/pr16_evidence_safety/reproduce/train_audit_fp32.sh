#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.audit_pr16_train --source /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pr16_evidence_safety_20261105/train_audit_fp32 --belief-precision fp32 > /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/pr16_evidence_safety_20261105/train_audit_fp32.log 2>&1
