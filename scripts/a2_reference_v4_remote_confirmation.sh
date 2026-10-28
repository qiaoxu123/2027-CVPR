#!/usr/bin/env bash
# Run only separate V4 post-training code; preserve active scientific source.
set -euo pipefail
TASK_ROOT=/mnt/hdd8t/Workspace/2027-CVPR-A2-20261028
LEGACY_ROOT="$TASK_ROOT/legacy_home/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBelief"
SOURCE_TREE="$TASK_ROOT/legacy_home/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBeliefV4"
GUEST_ROOT=/home/20261024_1/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBelief
case "${1:-run}" in
  cpu)
    A2_CUDA_DEVICE=''
    A2_MODULE=experiments.a2_waldo_landmark_belief.reference_v4_confirmation_preparation
    ;;
  run)
    A2_CUDA_DEVICE=0
    A2_MODULE=experiments.a2_waldo_landmark_belief.reference_v4_confirmation_pipeline
    ;;
  *) echo 'Usage: a2_reference_v4_remote_confirmation.sh cpu|run' >&2; exit 2 ;;
esac
exec env CUDA_VISIBLE_DEVICES="$A2_CUDA_DEVICE" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  HF_HOME=/home/20261024_1/.cache/huggingface HF_HUB_CACHE=/home/20261024_1/.cache/huggingface/hub \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  LD_LIBRARY_PATH="$TASK_ROOT/tools/root/usr/lib/x86_64-linux-gnu" \
  "$TASK_ROOT/tools/root/usr/bin/proot" -b "$TASK_ROOT/legacy_home:/home/20261024_1" \
  -b "$SOURCE_TREE:$GUEST_ROOT" -b "$LEGACY_ROOT/runs:$GUEST_ROOT/runs" -w "$GUEST_ROOT" \
  /home/20261024_1/miniconda3/envs/graphnav/bin/python -u -m "$A2_MODULE"
