#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105
while [ ! -f "$RUN/finish.exit" ]; do sleep 10; done
if [ "$(cat "$RUN/finish.exit")" != 0 ]; then echo 'Evaluation failed; reporting stopped'; exit 1; fi
/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -u -m scripts.report_unified_full_relation --run "$RUN" --data-root /home/20261024_1/Workspace/DATA/sbfnav-data --sbf-source /home/20261024_1/Workspace/2027-CVPR/Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source --group-inventory /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_fulltrain_s42_20261105/val_seen_inventory --output reports/unified_pgvb/full_relation_s42_20261105 > "$RUN/report.log" 2>&1
