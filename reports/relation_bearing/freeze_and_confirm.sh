#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-relation-bearing
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106
for SPEC in old matching relations angle_only/matching angle_only/relations; do
 case "$SPEC" in
  angle_only/*) SESSION="relation_bearing_angle_${SPEC#angle_only/}";;
  *) SESSION="relation_bearing_fit_$SPEC";;
 esac
 while ! "$PY" -c 'import json,sys;from pathlib import Path;p=Path(sys.argv[1]);sys.exit(0 if p.exists() and len(json.loads(p.read_text()))==12 else 1)' "$RUN/$SPEC/history.json"; do
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then echo "Incomplete run: $SPEC";exit 1;fi
  sleep 10
 done
done
"$PY" -m scripts.freeze_role_polygon_bearing --run "$RUN" > "$RUN/freeze.log" 2>&1
"$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]))["models"];assert d["relations_angle"]["dev"]["macro_Top1"]>d["old"]["dev"]["macro_Top1"],"No development gain; keep confirmation sealed"' "$RUN/FROZEN_BEFORE_CONFIRM.json"
"$PY" -u -m scripts.evaluate_role_polygon_bearing --run "$RUN" > "$RUN/confirmation.log" 2>&1
