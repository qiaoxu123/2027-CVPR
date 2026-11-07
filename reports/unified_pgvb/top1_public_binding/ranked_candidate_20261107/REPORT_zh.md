# 冻结候选排序头：完整 Val-Seen 闭环诊断

## 结论

在相同的 2,470 条 Val-Seen 任务、相同 Geometry/公开地标绑定/控制器 Checkpoint/Stop 规则下，冻结 Stage1 候选排序头每条任务只选择一次候选并保持，SR 为 **42.91%**，低于 Geometry Top-1 原策略的 **48.54%**（下降 **5.63 个百分点，净少139条成功**）。因此这项非视觉排序对照没有支持直接采用该候选排序头，也没有达到当前 55% SR 目标。

这不是视觉重排序结果：该实验不读取 RGB 作为排序证据，不改 Geometry 权重，也不训练权重。它用于确认只靠已有 Controller rank head 是否能将 Top-K 几何候选变成更好的固定导航目标。结果为否。

## 实验设置

- 运行分支：`experiment/v2-top1-public-binding-20261107`，代码基线 `c1902a96`。
- 配对任务：完整 Val-Seen 2,470/2,470；逐条 episode ID 完全一致。
- 两组均用扩展公开地标绑定、冻结 V2 Geometry、原 GraphNav Stage1 Controller、相同 Stop threshold 0.3、20-step Horizon、20m SR 判据、原动作与到达/高度守卫。
- 基线：几何 Top-1 + 原 `guarded-blend`。
- 对照：冻结 rank head 在 episode 开始时选择一个候选，之后 Goal 固定；路径和 Stop head 继续按当前位姿更新。无 RGB、无 GT 推理、无探索、无阈值调整。
- 权重 SHA256：Geometry `9ac51ee253eed46acbf66e6afb47888857b70258f5fcd195721723d5112dc387`；Controller `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`。

## 完整导航指标

| 方法 | SR | OSR | SPL | NE (m) | nDTW | 平均步数 | 路径长度 (m) | 单步推理 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Geometry Top-1 基线 | 48.54% | 55.43% | 47.15% | 30.83 | 0.299 | 6.81 | 196.17 | 6.41 |
| 冻结 Rank Head 一次选定 | 42.91% | 49.64% | 41.72% | 32.65 | 0.290 | 6.75 | 191.43 | 6.50 |
| 差值 | −5.63 pp | −5.79 pp | −5.43 pp | +1.82 | −0.009 | −0.06 | −4.74 | +0.09 |

配对变化：原失败→对照成功 **123** 条；原成功→对照失败 **262** 条；双方成功 **937** 条；双方失败 **1,148** 条。净变化为 −139 条成功，与总体 SR 差一致。rank head 选择非 slot-0 候选的任务有 **1,426** 条；在本协议下，改变候选并未带来足够可靠的导航收益。

## 与 RGB 证据诊断的关系

同目录上一阶段的冻结 SigLIP2 离线排序探针，在644条 Top-5 位置代理失败中产生57次位置代理纠正；在 Top-1 正确控制组产生103次位置代理破坏，合并净少46/1,842。Top-5失败组中310/644条有至少一个历史帧同时支持 Top-1 与替代候选的几何/深度 ROI；这只是观测支持，不等于人能或模型能辨认实例。人工审核网页的标签仍全部为待审核，**人工确认样本数为0**。

这些结果共同说明当前冻结视觉分数和已有 rank head 均不足以安全重排。没有可靠 RGB 证据时必须保留 Geometry Top-1。现阶段不应将该视觉规则或 rank-head 单次重选作为 V4 核心方案，也不应基于旧轨迹改 Goal 后继续使用非因果未来 RGB 来推算 SR。

## 产物

- `summary.json`：汇总、逐任务配对计数和输入协议。
- `paired_episode_outcomes.csv`：全部2,470条逐任务基线/对照指标与选中候选槽位。
- `selected_slot_counts.csv`：冻结 rank head 的候选槽位选择分布。
- `paired_navigation.png`：SR与配对纠正/破坏图。
- `../visual_candidate_reranking_20261107/review/index.html`：人工审核页。当前尚未收到人工标签；页面标注需由审核者导出后才能计入人工确认结果。
- 原始逐步闭环 rows 和 manifest 保存在服务器运行目录：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/v2_ranked_candidate_20261107/full_ranked/expanded/`。

## 决策

暂停扩大冻结视觉重排序实验。先取得人工确认的真实实例可辨认标签，并把监督目标从“20m位置代理”改为候选实例是否与指令目标匹配；之后仅在明确可靠、跨视图一致的观测上训练/校准候选排序器。任何在线切换还需采用因果闭环重新 rollout。当前论文/导航主方案保持 Geometry Top-1 原流程。
