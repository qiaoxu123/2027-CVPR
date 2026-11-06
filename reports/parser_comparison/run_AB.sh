#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-parser-benchmark
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/parser_comparison_s42_20261106
PY=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106/parser_env/bin/python
for P in A B; do
 "$PY" -u -m scripts.run_parser_benchmark --inputs "$RUN/data/public_inputs.json" --output "$RUN/$P" --parser "$P" > "$RUN/${P}.log" 2>&1
done
