# P2：可靠实例 RGB–Language Grounding

从 `4a69bdf` 独立开发。固定协议在 `protocol.json`，不扩大 Residual Gain，不训练 GAT、视觉 Backbone 或导航主干。

- `data.py`：重放已有真实访问轨迹到当前 cutoff。每个实例保留质量最佳的因果 ROI；当前 ROI 优先，历史补充上限 128 实例。复用命中缓存，其余用同一冻结 SigLIP2 编码。保存帧来源、step、实际相机和 CUDA/FP32 smoke。
- `dataset.py` / `relations.py`：原始目标身份、属性满足和无法判定分别标注。对象 ID 仅用于关联与损失；缺目标全 Unknown。同颜色同类必须有已观察命名参考物、语言关系与明确几何矛盾，才能作为弱负例。
- `model.py`：轻量 Projection + Word–Patch Attention + Context Reference Attention；Match/Mismatch/Unknown。正常输入不含目标身份、位置、地图名或静态 logits。City三组使用语言调制的已观察实例相对位置；RGB控制保持相同公开几何，原始目标位置不进入输入。
- `train.py`：每 family 一个确定的最新 prefix，3 Epoch，原始实例 R1 按 family 等权的 Train 内部 Dev 选 Epoch。AerialVG、RefSegRS 分别预训练，分别适配 CityNav，不混训。
- `external.py`：绑定 DIOR 官方 XML/对象计数读取顺序，仅冻结、标注 bbox proposal 条件下诊断，不把全球位置未知权重迁移到正式 CityNav。
- `evaluate.py`：Dev 温度与弃权校准、冻结配置、一次 Seen。Geometry/native/旧 Head/新 Head/实际 Black RGB/同类别 Shuffled/Identity Oracle 使用相同状态、候选、历史。正常 Residual<=0.025；未接受和未观察区域完全不变。
- `audit.py`：来源时序、地图/对象隔离、独立 family、同颜色困难负例和语言截断统计。

所有 CityNav 视觉语义/颜色监督均标弱监督，没有把几何有效当人工 RGB 真值。RefSegRS 是 mask-derived tile 多正例辅助任务，AerialVG 区域短语只作属性基线；`aerial_identity.py`依据官方固定 commit 的 `targets.boxes[0]` 和完整 caption，单独构建原目标身份监督。标注 bbox proposal 条件下的选择不冒充端到端检测。报告区分训练选择 Dev 与固定 Seen，并保留所有失败。

本机 Python：`/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-assets/remoteclip/.venv/bin/python`。运行设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1`，数据只读，结果 `runs/instance_grounding/`，新权重和缓存只留本地。CLI 拒绝覆盖产物。

```bash
$TASK_PY -m experiments.instance_grounding.data plan --cache "$CACHE" --output "$NEW_PLAN"
$TASK_PY -m experiments.instance_grounding.data encode --plan "$NEW_PLAN" --reuse "$OLD_FEATURES" --output "$NEW_FEATURES"
$TASK_PY -m experiments.instance_grounding.train --train "$TRAIN" --dev "$DEV" --initialize "$AERIAL_HEAD" --city --output "$NEW_HEAD"
$TASK_PY -m experiments.instance_grounding.evaluate calibrate --features "$DEV" --heads "$HEADS_JSON" --output "$CALIBRATION"
# 冻结代码、Checkpoint与配置SHA后执行一次正式 Seen。
$TASK_PY -m experiments.instance_grounding.evaluate evaluate --features "$SEEN" --config "$CALIBRATION/frozen_config.json" --output "$NEW_SEEN_RESULTS"
$TASK_PY -m pytest -q
```

所有门槛通过才考虑完整 Seen 闭环；未通过不运行 Unseen。实际命令与耗时记录会随报告保存，README 不将计划实验写成已运行结果。

实际固定Seen结论：选定City-only的family等权身份R1为18.68%，Native为19.56%；Real−Black +15.39pp、Real−Shuffled +12.19pp，但Goal R1仍44.53%、Recovery/Damage均0。仅36个独立几何可信可见错误family，三分类Unknown退化，全部联合门槛未通过；没有闭环或Unseen。外部Aerial完整指令Dev可由36.03%提高到60.34%，属于训练选择Dev及标注bbox条件任务，未证明CityNav迁移。完整报告与逐样本审计见 `docs/RELIABLE_INSTANCE_GROUNDING_REPORT_20261011.md`。
