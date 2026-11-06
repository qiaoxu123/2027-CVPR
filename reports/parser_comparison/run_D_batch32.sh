#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-parser-benchmark
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/parser_comparison_s42_20261106
PY="$RUN/gliner_env/bin/python"
"$PY" -u -m scripts.run_parser_benchmark --inputs "$RUN/data/public_inputs.json" --output "$RUN/smoke_D_batch32" --parser D --model /home/20261024_1/Workspace/models_local/Qwen3-14B --limit 32 --batch 32 > "$RUN/smoke_D_batch32.log" 2>&1
"$PY" -c 'import json,sys;rows=[json.loads(s) for s in open(sys.argv[1])];assert len(rows)==32 and not any(isinstance(x["raw"],dict) and "runtime_error" in x["raw"] for x in rows),"Inspect GPU batch32 failure"' "$RUN/smoke_D_batch32/predictions.jsonl"
"$PY" -u -m scripts.run_parser_benchmark --inputs "$RUN/data/public_inputs.json" --output "$RUN/D" --parser D --model /home/20261024_1/Workspace/models_local/Qwen3-14B --batch 32 > "$RUN/D.log" 2>&1
