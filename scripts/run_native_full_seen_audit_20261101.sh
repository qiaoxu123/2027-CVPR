#!/usr/bin/env bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Ablations/GraphNav/visual-residual-20261013
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
while [[ ! -f aerialvg_native_temporal_full_seen_paired_20261101.json ]]; do
 if ! pgrep -f '^bash scripts/run_native_temporal_full_seen_20261101.sh$' >/dev/null; then
  echo 'Full Seen runner ended without paired completion report; refusing success verification.' >&2
  exit 1
 fi
 sleep 10
done
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -m scripts.verify_native_navigation_result \
 --directory aerialvg_native_temporal_navigation_full_seen_20261101 \
 --frozen aerialvg_native_temporal_frozen_seen_candidate_20261101.json \
 --expected-episodes 2470 --require-full
