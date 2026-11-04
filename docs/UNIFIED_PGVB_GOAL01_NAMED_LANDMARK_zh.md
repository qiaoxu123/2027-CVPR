# Goal-01：先到达一个真实具名地标，再验证近场几何（UnifiedPGVB-v2）

**状态：代码已实现、CPU回归待确认；真实GPU闭环尚未由本分支执行。**
这是一个小目标和工程验收记录，**不是训练提示词**，也不允许直接宣称成功。

## 一句话目标

从已有冻结Val-Seen64地标清单中，**固定选取索引最小的Building任务**，保持原真实起点、原yaw、原协议。让无人机执行原文 `Navigate to <真实地标名称>.`：

1. **先验收导航到地标本身：** 进入该公开Polygon的20m范围，并正确停止。禁止依赖原始CityNav Target GT、Oracle Goal或提前在任务中筛选成功案例。
2. **再验收附近的公开几何：** 当UAV真实到达20m内时，自动记录其周边100m内的Building/Road/Parking/Ground等公开Polygon，与地标的边界距离、东/北方向向量、相交/包含关系，人工审查2–3例。该邻居表暂时不参与PGVB评分、控制或Stop。

如果第1项不成功，**不要把第2项当作已验收**；先从轨迹、Belief、Stop与RGB追踪日志定位失败机制。

## 当前设计与严格边界

- Geometry先引导。绝不因为估计距离<50m或Polygon已投影进入RGB就宣布视觉确认。
- 当前ROI来自公共Polygon+相机Footprint高精度几何投影，**不是真实像素分割真值或地标身份识别结果**。
- 用同一真实RGB的PGVB正/负可靠Patch对比，并与**同帧空间错位Patch特征**的反事实评分对照；连续2帧都通过才将模式由Geometry切换为RGB-supported Tracking。单次未验证不切换。3帧强证据后继续增加融合权重。短暂丢失可保留“coasting”状态，但不能加旧视觉证据。
- 当前只是经过约束的RGB支持追踪，**不是独立视觉地标身份验证**；尤其不能宣称仅凭图像读出了地标专有名称。时间阈值和判别阈值是初始工程设置，须通过TRAIN校准而不是在Val-Seen任务上反复挑最优。
- 不重训SigLIP2，不新增视觉网络，不改变已冻结Trajectory/Stop、horizon、20m规则；原Geometry和视觉评分都在一个UnifiedPGVB模型内。
- 新旧对照保持同任务、同权重、同原始RGB/yaw：`Geometry`（不提取RGB），`LegacyPGVB`（原固定残差），`TrackedPGVB`（新门控）。
- 实时诊断中的公共Polygon距离只用于分析，不参与模式切换、行动选择或停止条件。

## 本地AI在GPU服务器上的最小执行目标

拉取本分支：
`experiment/unified-pgvb-landmark-acquisition-20261104`。

先运行CPU测试并确认V2训练Pilot权重可以被最新运行时完整加载：

```bash
python -m pytest -q tests/test_landmark_acquisition.py tests/test_landmark_neighborhood.py \
  tests/test_unified_instruction.py tests/test_unified_pgvb.py tests/test_unified_pgvb_v2.py
```

从**真实** `tasks_public.json`中确定最小索引Building；脚本不能按SR挑任务：

```bash
export GOAL_TASK_INDEX="$(python - <<'PY'
import json, os
from pathlib import Path
tasks=json.loads((Path(os.environ["SELECTION_DIR"])/"tasks_public.json").read_text())
choice=min((t for t in tasks if t["category"]=="Building"), key=lambda t:t["index"])
print(choice["index"])
print(f'GOAL={choice["instruction"]} on {choice["map_name"]}', file=__import__("sys").stderr)
PY
)"
```

用固定64清单、已核验的Stage1和V2 Pilot `full_training/last.pt`（带 `unified-pgvb-v2` 格式）运行**同一条**：

```bash
python -m scripts.run_unified_pgvb_landmark_closedloop \
  --selection "$SELECTION_DIR" \
  --checkpoint "$STAGE1_CHECKPOINT" \
  --unified-checkpoint "$V2_CHECKPOINT" \
  --sbf-source "$SBF_SOURCE" \
  --config "$SBF_CONFIG" \
  --data-root "$CITYNAV_ROOT" \
  --output "$GOAL01_OUTPUT_DIR" \
  --methods Geometry LegacyPGVB TrackedPGVB \
  --goal-task-index "$GOAL_TASK_INDEX"
```

输出 `predictions.jsonl` / `metrics.json` / `decisions/*/step_*.npz` 逐步审查：
- 起点真实RGB有效性、Public ROI投影、patch正负样本数。
- 从Geometry→candidate→tracking发生的具体帧、视觉AUC与错位对照AUC，是否真正发生切换。
- 若未切换，判断是从未进入视野、缺少可靠负Patch、同帧PGVB分数不辨识，还是时序确认不足。**不能通过去掉反事实对照和确认条件伪造追踪成功。**
- Geometry/Legacy/Tracked三版分别SR、OSR、NE、首入20m时刻、正确Policy Stop及运行时长。
- 首次进入20m时的 `near_landmark_geometry` 与邻居表（最多12个公共实体）的类型、相对dx/dy、边界距离和相交/包含关系，核对位置方向。
- 检查名称解析、Pose→Footprint→RGB与世界坐标的朝向、Polygon投影稳定性；yaw不准时必须在报告中举证。
- 失败时只修复有证据的**几何投影、推理接线、状态管理或控制器软件缺陷**；不要修改成功阈值、Oracle输入、任务样本、预训练权重或原评测协议以获得1/1。回归通过后用同一固定任务复测并保留旧结果。

**目标验收**：至少报告一条真实RGB闭环的完整因果链：
`Navigate to X → Geometry接近 → RGB判别/追踪（或诚实未触发） → 最终GoalField → 进入20m → Stop → 公共邻居几何核验`。

如果1/1失败，交付轨迹与明确的失败原因和下一项最小修复，不因未成功而更换任务或宣称成功。

## 后续

完成Goal-01后，再使用固定64**全部任务**比较三种模式SR/OSR/Stop（Other与无视觉证据任务仍计分母）。不得用一条成功替代64条总指标，也不得将具名Polygon代理SR冒充CityNav原始Target SR。确认Geometry能有效接近地标、视觉采集/追踪可信后，再进入真实关系目标Belief训练。

结果保存在 `reports/unified_pgvb/landmark_acquisition/GOAL01_REPORT_zh.md`，保存真实任务索引、原始命令、所用Checkpoint SHA、时间与跟踪证据。所有修改独立分支Commit/Push，不覆盖main或已完成Pilot。
