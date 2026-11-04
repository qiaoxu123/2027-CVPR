#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.capture_named_training_views \
 --dataset-profile configs/datasets/landmark_associated_train_v1.json \
 --map-manifest /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/landmark_chain_visual/chain_train400_features/manifest.json \
 --catalog /home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json \
 --data-root /home/20261024_1/Workspace/DATA/sbfnav-data \
 --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source \
 --backbone /home/20261024_1/.cache/huggingface/hub/models--google--siglip2-large-patch16-384/snapshots/1b426889ea62b5a72bf9839009a1b184bfc9c178 \
 --output /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/geometry_guided_visual_belief/public_training_views_v1 \
 > /tmp/geometry_guided_visual_belief/capture.log 2>&1
printf '0\n' > /tmp/geometry_guided_visual_belief/capture.exit
