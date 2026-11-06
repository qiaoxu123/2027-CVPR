# CityNav v5 independent retrospective verification

[中文报告](REPORT.md) · [冻结协议](PROTOCOL.md) · [外部审查记录](REVIEW.md)

基于v4提交 `6ae8b68`，独立分支 `experiment/citynav-frame-generalization-v5-20261106`。TRAIN-only，不读取Val/Test，不训练或修改任何导航/语言图模型。

## 复跑

在仓库根目录运行，使用既有conda环境Python。每步必须退出0才进入下一步；失败后不得使用旧产物继续统计。

```bash
set -e
PY=/home/20261024_1/miniconda3/envs/graphnav/bin/python
$PY experiments/direction_frames_v5/freeze_v4.py
$PY -m pytest -q experiments/direction_frames_v5/test_v5.py
$PY experiments/direction_frames_v5/prepare_v5.py
$PY experiments/direction_frames_v5/explore_v5.py
# 以上所有角色、场与对照在读取新GT之前冻结：
$PY experiments/direction_frames_v5/score_v5.py
$PY experiments/direction_frames_v5/audit_v4.py
$PY experiments/direction_frames_v5/supplement_v5.py
$PY experiments/direction_frames_v5/geometry_diagnostics_v5.py
$PY experiments/direction_frames_v5/visualize_v5.py
$PY experiments/direction_frames_v5/report_v5.py
$PY -m pytest -q experiments/direction_frames_v5/test_v5.py > experiments/direction_frames_v5/results/test_results.txt
$PY experiments/direction_frames_v5/verify_v5.py
```

数据路径继承v4 `shared.py` 的DATA，原始TRAIN和CityRefer对象SHA继承v4数据manifest。公开大数据不复制入Git。gzip的压缩时间戳可改变字节SHA；对应manifest会更新，逐行预测/统计应一致。没有安装新依赖或加载新预训练模型。

## 文件说明

以下数据文件位于 `results/`：

- `v4_lock.json`：v4全部Python源码、原4701账本和冻结结果/输入SHA。验证时另以Git基线检查继承v2/v3工具依赖。
- `semantic_scan.jsonl.gz`：全TRAIN搜索记录、原文、所有具名角色候选与方向span。`requested_structure_inventory.csv/json`分开粗关键词匹配与真正具名模板匹配。
- `new_candidates.jsonl.gz`、`new_relation_audit.csv/json`：44个新增/复查记录，包含39个新ID、5个旧ID；主语作用域、A/B道路ID/span、分值、排除原因、模板与新颖性、相对于v4唯一开发集的地图/道路/主语资格。
- `public_inputs.jsonl.gz`：原1398输入完全复用，加15个新可靠绑定；不含目标位置/轮廓、yaw或未来轨迹，场景剔除目标ID。
- `new_frames.jsonl.gz`：新语义适配后的逐关系输出与证据；全部原4701冻结预测继续保存在v4，未被覆盖。`all_relation_outputs.jsonl.gz`记录4740条最终状态，包括所有失败。
- `road_fields.jsonl.gz`：冻结v4算法构造的完整场；`source_topology_audit.csv/json`：全部端点/来源接触及连通性；`resolution_fields.jsonl.gz`：预先生成的0.5/0.75/1.25/2m几何诊断。
- `exploratory_fields.jsonl.gz`：F统一平滑方案，未逐例选最优；`negative_fields.jsonl.gz`：GT前构造的错误来源/错误参照物完整场或unknown。
- `phase1_manifest.json`：输入、解析器、完整场、探索与负对照的冻结SHA；`evaluation_manifest.json`：GT、评价代码与阶段关联。
- `evaluation_gt.jsonl.gz`：仅离线GT账本。`geometry_checks.csv/json/jsonl.gz`：全部4740条状态/侧别结果。
- `summary.csv/json`：完整与高置信的覆盖/可判定/条件率/总分母确认率、Wilson与对象/道路/地图聚类CI。
- `grouped_results.csv/json`：地图、道路、句式、对象类别、词的全部分项，包括空支持/失败。
- `method_predictions.jsonl.gz`：A–F、四世界轴、留图/留路世界字典与负对照逐条结果；`method_comparisons.csv/json`：同支持净差及聚类CI。A–E严格复用同中心线投影/mask/完整轮廓谓词，F单独改变几何。
- `deduplicated.csv/json`：同对象/道路/词/场去重；不是把语言相似度直接当对象重复。
- `frozen_binding_support.csv/json`：冻结v4绑定函数是否能产生同一关系；`frozen_language_comparison.csv/json`：给定新绑定后原v4参考方向模块的输出。
- `v4_all44_reaudit.csv/json`：原44条完整重算；`v4_ambiguous22_classification.csv/json`：所有22条歧义的实际触发和未证实归因、点诊断与容差/分辨率结果。
- `Abbey_failure_analysis.json`：原文、全部来源分段、拓扑、其他描述、真实侧别冲突及结论边界；方向从未按GT翻转。
- `resolution_sensitivity.csv/json`、`F_geometry_transitions.csv/json`：几何变化逐条记录；`random_rotation_control.json`：300次同mask随机旋转对照。
- `case_index.json`与 `figures/`：真实TRAIN地图案例及统计图，320dpi PNG/矢量PDF。GT只用于离线检验与图示，构造单元测试不进入真实结果。
- `road_geometry_features.csv/json`、`geometry_strata.csv/json`：宽度净空代理/弯曲程度的冻结后描述性分组，不用于调参。
- `dataset_source_manifest.json`：原TRAIN/CityRefer输入SHA与split范围。
- `verification.json`、`dependency_lock.json`、`test_results.txt`：最终复算与源文件校验；`adversarial_review*`：实际插件调用结果，不等同于审查通过。

“唯一道路场”只是在声明的2D地图提取假设下有向，不是已确认作者yaw。所有泛化结论均为TRAIN内回顾性，Val/Test未读取。
