#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-parser-benchmark
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/parser_comparison_s42_20261106
"$RUN/gliner_env/bin/python" -u -m scripts.run_parser_benchmark --inputs "$RUN/data/public_inputs.json" --output "$RUN/C" --parser C --model /home/20261024_1/Workspace/models_local/gliner2-base-v1 > "$RUN/C.log" 2>&1
