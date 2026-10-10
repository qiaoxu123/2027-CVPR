# Visual-to-Goal Association / Observation Coverage

基于 `350f99d` 的独立 P0/P1 实验，不训练 Backbone、GAT 或导航模型。协议见 `protocol.json`；正式结果见 `docs/VISUAL_GOAL_ASSOCIATION_REPORT_20261011.md`。

- `association.py`：原 centroid、实际已观察 Polygon 与 Grid 的面积交叠、有限距离软关联、实例聚合再 max 传播。所有实例支持先裁剪到实际相机 footprint；未关联区域残差严格为0。
- `evaluate.py`：冻结已有 AerialVG+CityNav Projection Head，复用 Real/Zero/Shuffled 缓存；Train 内部 Dev 的256个配置只按 Goal 指标选择，冻结后 Seen 单次验证。Oracle 身份位于独立评估分支，不读取目标坐标决定关联或动作。主 Oracle 与所有 RGB 使用相同有效特征实例；`all_cached_rois` 是单独标记的覆盖上界，不能与主对照混为一谈。
- `coverage.py`：当前、稀疏缓存历史、全部因果已访问人类轨迹历史，及每状态最多2个合法额外观察。Approach / Descend / Lateral 使用静态预测 Top1；belief-active 使用原有公开 Top8 不确定性与已观察置信度。政策函数没有 GT 参数。新增 RGB/Depth 均实际渲染在所访问模拟位置，未编码为新视觉特征，也没有导航 rollout。
- `audit.py`：只分析已计算分数/排序/冲突/尺度，目标身份仅在离线结果统计中出现。

统计单位是640个正向 prefix、128 families。报告同时给出独立对象及 family 数，地图及 family 配对 Bootstrap，固定 Seed20261011。稀疏历史仅含已缓存的 prefix；dense causal history 只读取当前 cutoff 之前真实人类访问的位置，记录重放成本。深度是已有 CityNav 垂直高度代理，不代表遮挡检测；几何有效和 depth>=0.9 不代表人工语义真值。

本机实际运行关键命令（已有产物不可覆盖；复现请换新输出目录）：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
TASK_PY=/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-assets/remoteclip/.venv/bin/python
TASK_OLD=/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-main/runs/bidirectional_core_test
TASK_RUN=runs/visual_goal_association
TASK_RGB=runs/rgb_language_grounding
$TASK_PY -m experiments.visual_goal_association.evaluate prepare --cache "$TASK_RGB/cache_internal_dev" --features "$TASK_RGB/features_city_dev_public" --controls "$TASK_RGB/controls_city_dev" --checkpoint "$TASK_RGB/heads/aerial_city_adapt/best_dev.pt" --output "$TASK_RUN/prepared_dev"
$TASK_PY -m experiments.visual_goal_association.evaluate calibrate --prepared "$TASK_RUN/prepared_dev" --output "$TASK_RUN/calibration_dev_verified"
$TASK_PY -m experiments.visual_goal_association.evaluate evaluate --prepared "$TASK_RUN/prepared_dev" --config "$TASK_RUN/calibration_dev_verified/frozen_config.json" --output "$TASK_RUN/evaluation_dev"
$TASK_PY -m experiments.visual_goal_association.coverage --cache "$TASK_RGB/cache_internal_dev" --output "$TASK_RUN/coverage_dev"
# 配置/源代码/测试的SHA冻结于 freeze.json 后才启动 Seen。
$TASK_PY -m experiments.visual_goal_association.evaluate prepare --cache "$TASK_OLD/cache_fp16/val_seen" --features "$TASK_RGB/features_city_seen_public" --controls "$TASK_RGB/controls_city_seen" --checkpoint "$TASK_RGB/heads/aerial_city_adapt/best_dev.pt" --output "$TASK_RUN/prepared_seen"
$TASK_PY -m experiments.visual_goal_association.evaluate evaluate --prepared "$TASK_RUN/prepared_seen" --config "$TASK_RUN/calibration_dev_verified/frozen_config.json" --output "$TASK_RUN/evaluation_seen"
$TASK_PY -m experiments.visual_goal_association.coverage --cache "$TASK_OLD/cache_fp16/val_seen" --output "$TASK_RUN/coverage_seen"
$TASK_PY -m experiments.visual_goal_association.audit --prepared "$TASK_RUN/prepared_seen" --output "$TASK_RUN/association_audit_seen"
$TASK_PY -m pytest -q
```

主干权重/大缓存保留本地；运行目录包含原始逐样本 JSONL、实际观察子集和失败日志。小型结果与压缩逐样本证据随报告保存。禁止把观察覆盖率当作目标理解准确率，禁止把 Oracle 当作可部署结果。
