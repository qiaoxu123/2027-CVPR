#!/usr/bin/env bash
set -euo pipefail
R=/mnt/hdd8t/Workspace/2027-CVPR-A2-20261028
GUEST_ROOT=/home/20261024_1/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBelief
SOURCE_TREE=$R/legacy_home/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBeliefV4
OLD_TREE=$R/legacy_home/Workspace/2027-CVPR/Experiments/A2WaldoLandmarkBelief
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HOME=/home/20261024_1/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH=$R/tools/root/usr/lib/x86_64-linux-gnu
cpu_python() {
    "$R/tools/root/usr/bin/proot" \
        -b "$R/legacy_home:/home/20261024_1" \
        -b "$SOURCE_TREE:$GUEST_ROOT" \
        -b "$OLD_TREE/runs:$GUEST_ROOT/runs" -w "$GUEST_ROOT" \
        /home/20261024_1/miniconda3/envs/graphnav/bin/python -u \
        -m diagnostics.evidence_quality.pipeline "$@"
}
cpu_python --prepare-cpu
cpu_python --wait
