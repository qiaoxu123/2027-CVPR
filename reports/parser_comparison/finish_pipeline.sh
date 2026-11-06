#!/bin/bash
set -euo pipefail
cd /home/20261024_1/Workspace/2027-CVPR-parser-benchmark
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/parser_comparison_s42_20261106
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
TESTPY=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106/parser_env/bin/python
while [ ! -f "$RUN/D/runtime.json" ]; do
 if ! tmux has-session -t parser_comparison_D 2>/dev/null; then echo 'D stopped before complete runtime manifest';exit 1;fi
 sleep 10
done
"$PY" -c 'import json,sys;from pathlib import Path;r=Path(sys.argv[1]);rows=[json.loads(s) for s in (r/"D/predictions.jsonl").read_text().splitlines()];assert len(rows)==len({x["index"] for x in rows})==400;assert json.load(open(r/"D/config.json"))["inputs_SHA"]=="fc9821b1d679b3299873b6e74ee6fd03c46ff6052bce355acd23bb0ebfc3d376"' "$RUN"
"$PY" -m scripts.normalize_qwen_parser_spans --run "$RUN" > "$RUN/D_span_normalization.log" 2>&1
"$PY" -m scripts.report_parser_benchmark --run "$RUN" --output reports/parser_comparison > "$RUN/final_summary.log" 2>&1
"$PY" -m scripts.build_parser_review --run "$RUN" --output reports/parser_comparison
"$PY" -m scripts.build_parser_comparison_cases --report reports/parser_comparison
"$PY" -m scripts.write_parser_report_zh --report reports/parser_comparison
"$TESTPY" -m pytest -q > reports/parser_comparison/CPU_tests.txt 2>&1
"$TESTPY" -m scripts.smoke_parser_review_ui --html reports/parser_comparison/review.html --output reports/parser_comparison/UI_smoke.json > "$RUN/UI_final.log" 2>&1
"$PY" -c 'import json;from pathlib import Path;Path("reports/parser_comparison/FINISH_STATUS.json").write_text(json.dumps(dict(all_four_parsers_400=True,model_direction_training=False,formal_Val_Test=False,independent_human_gold_count=0,CPU_and_UI_passed=True),indent=2))'
echo 'ALL INFERENCE, POSTPROCESSING, REPORTS AND TESTS COMPLETE; inspect before commit'
