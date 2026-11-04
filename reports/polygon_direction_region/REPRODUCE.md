# 冻结实验复现

在`Experiments/GeomBeliefLandmarkChain`执行；使用现有graphnav环境，独立输出目录，保留已封存报告。

```bash
TASK_PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
TASK_DATA=/home/20261024_1/Workspace/DATA/sbfnav-data
TASK_MODELS=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/direction_belief_v1/pilot_s0
TASK_OUT=runs/polygon_direction_region_reproduction
TASK_PRIVATE=runs/polygon_direction_region_private_reproduction

"$TASK_PY" -m scripts.scan_polygon_direction --structured reports/pgvb/language/structured_train.json.gz --catalog "$TASK_DATA/cityrefer_original/objects.json" --output "$TASK_OUT"
"$TASK_PY" -m scripts.select_polygon_direction --directory "$TASK_OUT" --old-public reports/direction_belief/data_split/public.json
"$TASK_PY" -m scripts.prepare_polygon_direction --directory "$TASK_OUT" --catalog "$TASK_DATA/cityrefer_original/objects.json" --annotations "$TASK_DATA/original_citynav/citynav_train_seen.json" --data-root "$TASK_DATA" --old-public reports/direction_belief/data_split/public.json
"$TASK_PY" -m scripts.evaluate_polygon_direction --directory "$TASK_OUT" --models "$TASK_MODELS"
"$TASK_PY" -m scripts.export_polygon_direction_labels --directory "$TASK_OUT" --annotations "$TASK_DATA/original_citynav/citynav_train_seen.json" --structured reports/pgvb/language/structured_train.json.gz --subset fit_dev --output "$TASK_PRIVATE/fit_dev.json"
"$TASK_PY" -m scripts.score_polygon_direction --directory "$TASK_OUT" --labels "$TASK_PRIVATE/fit_dev.json" --phase development
```

检查`development_decision.json`。本次测得门槛失败，没有训练步骤。只有冻结方案后，才可一次性评分确认组：

```bash
"$TASK_PY" -m scripts.export_polygon_direction_labels --directory "$TASK_OUT" --annotations "$TASK_DATA/original_citynav/citynav_train_seen.json" --structured reports/pgvb/language/structured_train.json.gz --subset confirm --output "$TASK_PRIVATE/confirm.json"
"$TASK_PY" -m scripts.score_polygon_direction --directory "$TASK_OUT" --labels "$TASK_PRIVATE/confirm.json" --phase confirmation
"$TASK_PY" -m scripts.report_polygon_direction --directory "$TASK_OUT" --data-root "$TASK_DATA"
"$TASK_PY" -m scripts.summarize_polygon_direction --directory "$TASK_OUT"
"$TASK_PY" -m pytest -q tests/test_polygon_direction_region.py tests/test_direction_belief.py tests/test_direction_frame_audit.py tests/test_strict_direct_direction.py
```

复现固定方案不等于用同一确认集选择新方案。修改筛选、坐标、区域公式或参数后，不能继续将这30条称为新的独立确认组。若其他版本的开发门槛通过，应先另行实现/验证拟合流程并冻结模型，不能越过确认集导出保护。

SHA记录封存文件、脚本和Checkpoint的实际字节；重新打包NPZ时容器实现可能影响文件字节，数值复现应另比较概率数组与指标。稀有east/west例不构成独立验证。
