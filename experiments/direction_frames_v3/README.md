# CityNav local reference frames v3

中文结论与全部真实图表：[REPORT.md](REPORT.md)。基于 v2 `a942369`，仅原始 TRAIN；仅具名地标参照关系。

## 复跑

在仓库根目录执行。Python 必须来自已有 conda 环境；`prepare.py` 还需要 spaCy 3.7/en_core_web_sm（本机 parser venv 继承 graphnav site-packages）。原始数据不随 Git 提交，精确路径与 SHA256 见 `results/dataset.json`。

```bash
PY=/home/20261024_1/miniconda3/envs/graphnav/bin/python
PARSER=/tmp/citynav_direction_parser_env/bin/python
ROOT=/home/20261024_1/Workspace/DATA/sbfnav-data
$PARSER experiments/direction_frames_v3/prepare.py --data-root "$ROOT"
$PY experiments/direction_frames_v3/infer.py
$PY experiments/direction_frames_v3/embed.py --device cuda
$PY experiments/direction_frames_v3/evaluate.py
$PY experiments/direction_frames_v3/finish_analysis.py
$PY experiments/direction_frames_v3/render.py
$PY experiments/direction_frames_v3/report.py
$PY -m pytest -q experiments/direction_frames_v3/test_rules.py
$PY experiments/direction_frames_v3/verify.py
```

BERT 使用本地缓存快照 `86b5e0934494bd15c9632b12f734a8a67f723594`，`local_files_only=True`。若数据、模型路径不同，在显式参数/`embed.py` 中调整后重跑；CPU 可传 `--device cpu`。gzip 文件压缩时间戳可改变文件 SHA，manifest 会随复跑自动更新，行内容预测应一致。GPU 数值可有细微差别。

## 产物

- `relations.jsonl.gz`：全部4,659条具名地标候选，主语/参照物、作用域、解析置信度、原文和排除原因。
- `binding_counts.csv/json`, `exclusion_reasons.csv/json`, `binding_expansion.csv`：覆盖率、分词排除原因、新增307条类型消歧的逐条绑定。
- `public_inputs.jsonl.gz`：推断白名单输入，关系主语坐标/轮廓已剔除，包含起点但无yaw/未来轨迹。主语ID仅用于账本对齐，不是BERT/分类器特征。
- `evaluation_gt.jsonl.gz`：GT仅供离线评价；与推断输入分离。
- `inference_predictions.jsonl.gz`, `all_frame_candidates.csv`, `routing_rules.csv`：先于GT评价冻结的候选方向、独立证据、unknown与固定选择规则。
- `reference_frame_outputs.jsonl.gz`：全部4,659条输出，含未绑定unknown、参考系/候选朝向/状态/置信度。silver预测概率不代表作者参考系置信度。
- `bert_features.npz`, `encoder_texts.jsonl.gz`, `bert_feature_manifest.json`：冻结BERT语义特征与本地权重SHA。
- `silver_router_full_train.npz`, `silver_router_model.json`：全部TRAIN银标签轻量重拟合；未参与报告CV，参数为scaler mean/scale及logistic coef/intercept/classes。
- `folds.jsonl.gz`, `router_predictions.csv`, `router_quality_silver.json`：对象折/逐地图划分、原英文重复剔除、银标签预测质量。
- `evaluation_predictions.csv/jsonl.gz`：每条关系、每种方法/验证/队列的固定候选质量、角误差、世界回退、成功值；CSV省略权重字典，JSONL包含完整权重。
- `summary.csv/json`, `breakdown.csv/json`, `subject_category_breakdown.csv/json`：条件兼容度、总体hybrid、coverage、unknown、配对净改善、对象/参照物/地图聚类CI；地图/方向词/解析置信度/语义类别与主语类别分项。
- `ablation_differences.csv/json`, `negative_controls.csv/json`：同支持均匀朝向消融、道路朝向先验消融、BERT对规则净差、旋转/置换与标签打乱。
- `case_index.json`, `strict_case_details.json`：原始英文、真实Polygon ID、严格方向案例、船屋已知开发例标记。
- `figures/`：15组320dpi PNG/矢量PDF，8个真实地图案例、7幅统计图。无生成图片或虚构统计。
- `verification.json`, `test_results.txt`：数据/预测/交叉验证/汇总与无GT选方向的不变量检查；`REVIEW.md`：实际外部审查执行状态。

所有上述JSON/CSV/NPZ账本位于 `results/`。原始地图只在8个案例中裁剪展示，未提交全量地图或预训练模型。完整地理方向未校准，不把地图+Y称为真实北方。
