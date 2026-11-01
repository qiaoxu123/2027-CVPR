#!/usr/bin/env bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
native_followup_python=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
while [[ ! -f aerialvg_native_onpolicy_train128_20261101/collection.json ]]; do
 if ! ps -p 524231 -o args= | rg -q 'scripts.collect_native_onpolicy'; then
  echo 'Collection process ended without completion manifest; refusing follow-up.' >&2
  exit 1
 fi
 sleep 5
done
"$native_followup_python" -m scripts.analyze_native_onpolicy --collection aerialvg_native_onpolicy_train128_20261101 --output aerialvg_native_onpolicy_train128_diagnostic_20261101.json
"$native_followup_python" -m scripts.diagnose_native_temporal_belief --collection aerialvg_native_onpolicy_train128_20261101 --checkpoint aerialvg_native_plain_sigma8_s0_20261101/best.pt --output aerialvg_native_temporal_train128_20261101.json
"$native_followup_python" -m scripts.prepare_native_onpolicy_training --teacher aerialvg_native_training_train256_20261101 --collection aerialvg_native_onpolicy_train128_20261101 --output aerialvg_native_training_onpolicy128_20261101
"$native_followup_python" -u -m scripts.train_object_query_belief --train aerialvg_native_training_onpolicy128_20261101 --dev aerialvg_native_training_dev128_20261101 --output aerialvg_native_plain_onpolicy128_sigma8_s0_20261101 --epochs 20 --kernel-sigma 8 --gain-lr .002 --seed 0
