# CityNav semantic reference frames v4

[中文报告](REPORT.md) · [执行计划](../../docs/superpowers/plans/2026-11-06-citynav-semantic-frames.md) · [外部审查真实状态](REVIEW.md)

基于v3 `7f1bfb8`，分支 `experiment/citynav-semantic-frames-20261106`。仅完整原始TRAIN与公开地图；没有训练或修改任何导航模型。

## 分阶段复跑

在仓库根目录执行，使用已有conda环境的Python；无新依赖安装。准备阶段不用spaCy/BERT，只复用已有账本并应用独立文本规则。所有依赖均为本机现有 NumPy/Shapely/SciPy/OpenCV/Pandas/Matplotlib/Rasterio/Pillow。

```bash
PY=/home/20261024_1/miniconda3/envs/graphnav/bin/python
$PY experiments/direction_frames_v4/audit.py
$PY experiments/direction_frames_v4/frames.py
# 必须确认两步成功且freeze_manifest.json更新，再进入GT阶段：
$PY experiments/direction_frames_v4/make_gt.py
$PY experiments/direction_frames_v4/evaluate.py
$PY experiments/direction_frames_v4/controls.py
$PY experiments/direction_frames_v4/final_analysis.py
$PY experiments/direction_frames_v4/render.py
$PY experiments/direction_frames_v4/report.py
$PY -m pytest -q experiments/direction_frames_v4/test_frames.py
$PY experiments/direction_frames_v4/verify.py
```

原始数据路径在 `shared.py` 的DATA设置，文件SHA在 `results/dataset.json`；不同机器可改明确的数据根目录后重跑。不要改方向规则以迎合GT。gzip压缩时间戳会使字节SHA改变，程序同步更新manifest；逐行预测内容应可复现。

## 输入与输出边界

`frames.derive(public_input, field_cache)` 不接收目标坐标/轮廓、未来轨迹、GT或yaw。仅使用公开白名单场景。返回逐关系类型、原点区域、范围/候选/空间道路场、文本span与geometry IDs，区分`unique`、`unique_field`、`multiple`、`unknown`。

`unique_field`是沿道路变化的有向场，不是已知目标位置下挑出的最佳yaw。`road_fields.jsonl.gz`冻结全部采样点、正向、连接边、原点和多分辨率敏感性；GT查询只存在于评价阶段，不能改方向场。唯一性是在明确地图提取假设下成立，非已验证注释者视角。

## 主要产物（均在results/，除图与报告）

- `train_semantic_scan.jsonl.gz`：完整TRAIN关键词检索，包括未绑定/不合格原文；`semantic_search_coverage.csv/json`：词汇结构覆盖。
- `relations.jsonl.gz`：全部4,701候选，含原v3全部关系及新增具名道路锚；`binding_extensions.csv/json`记录49个新增绑定（7旧作用域+42新语法）。原v3账本在上级版本目录保留。
- `public_inputs.jsonl.gz`：白名单公开输入，剔除主语及导航目标几何；ID仅用于账本对齐。
- `frozen_frames.jsonl.gz`、`frame_evidence.csv`：全部逐关系预测与unknown、依据span及geometry IDs；`road_fields.jsonl.gz`：可变方向场的完整定义。
- `freeze_manifest.json`：第一阶段代码、输入、场SHA；`GT_phase_manifest.json`：第二阶段读取GT前的冻结指纹；`evaluation_manifest.json`：评价代码和数据指纹。
- `evaluation_gt.jsonl.gz`：仅离线GT账本；`geometry_checks.csv/jsonl.gz`：逐条几何结果、谓词、margin和歧义原因。
- `summary.csv/json`：完整/高置信队列的唯一率、多解率、unknown率、条件率、整体确认率、Wilson与对象/地标/地图聚类CI。
- `breakdown.csv/json`、`word_counts.csv/json`、`frame_exclusion_reasons.csv`：逐词、地图、参照物类别、绑定来源与状态。
- `leave_map_out.csv/json`：同一无训练规则逐图应用；`matched_baselines.csv/json`、`matched_baseline_summary.json`：同44条支持世界/yaw/起点连线对照及配对净差。
- `negative_controls.csv/json`、`negative_control_details.jsonl.gz`：300次旋转、180翻转、错绑定、A/B角色交换；unknown不从原支持中删去。
- `fixed_geometry_language_interventions.json`、`fixed_origin_view_interventions.json`：真实地图几何上的构造语言干预，单独标记，不能混入自然样本统计。
- `shared_view_constraints.json`：非目标具名地标间约束角集合；不冒充精确作者yaw。
- `same_landmark_instructions.csv/json`、`opposite_relation_pairs.csv/json`：同地标与正反词逐对审计。
- `v3_strict_support_audit.csv/json`、`deduplicated_sensitivity.json`、`historical_context.json`：旧支持逐条比较、重复关系去重、v1/v2/v3原结果与SHA。
- `case_index.json`：8个真实案例的原文、绑定、证据、状态、结果；`figures/`包含8案例+6统计图的320dpi PNG与矢量PDF。
- `raster_geometry_manifest.json`：24张TRAIN地图的独立affine/CRS核查；`software.json`：本轮软件版本。
- `verification.json`、`test_results.txt`：最终复核与测试；`adversarial_review*`：实际外部审查记录。

GT只用于检验和图示，不用于绑定/朝向选择。构造语言只用于机制负对照，图表自然案例与结果均来自真实TRAIN。真实北方没有CRS校准；XY仅为原地图坐标。
