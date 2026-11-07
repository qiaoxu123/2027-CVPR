# Geometry 保真微调 Checkpoint 的128条闭环 Pilot

日期：2026-11-07。完整 Val-Seen 当前冻结基线：SR 48.54%（2470条）。本 Pilot 使用同一固定 Val-Seen 前128条，与此前 Geometry slot-0 的同任务 Pilot 配对；目标 GT 仅在 rollout 之后评分。

## 结果

| 指标 | V2 Geometry 基线 | TRAIN 保真微调 Geometry | 差值 |
|---|---:|---:|---:|
| SR | 53.91% (69/128) | 56.25% (72/128) | +2.34 pp (+3条) |
| OSR | 57.81% | 60.94% | +3.13 pp |
| SPL | 52.75% | 54.94% | +2.19 pp |
| NE | 24.98 m | 25.20 m | +0.21 m |
| nDTW | 0.2934 | 0.2997 | +0.0063 |
| 平均步数 | 7.39 | 7.38 | -0.02 |
| 推理时间 | 5.51 ms | 5.44 ms | -0.08 ms |

配对结果：失败→成功5条，成功→失败2条，净增3条。NE略变差，因此不能把各指标概括为全面改善。Pilot 完整覆盖128条，推理终止均为 `policy_stop`，没有改 Stop/Horizon/成功判据。

## 评测门槛与决策

用户设定的筛选门槛为 Pilot SR≥70% 才考虑完整评测。本组 56.25%，未达门槛，**未启动全量评测**。128条正向差异仅是小 Pilot 信号，不能证明完整 Val-Seen SR 会提升，也不能证明达到论文目标。

Checkpoint 来自 TRAIN 保真微调，选择基于 Cambridge TRAIN development；对应 TRAIN development R@20 为70.06%（源模型69.39%）。此定位指标与闭环 SR 不同。完整 Val-Seen 的初始定位旧结果中，保真微调仅由49.84%到49.96%，说明定位增益很小；本 Pilot 的SR变化可能受有限样本波动影响。

## 复现文件

- `manifest.json`：任务数、协议、seed、模型与代码哈希。
- `metrics.json`：128条聚合指标。
- `paired_episode_outcomes.csv`：逐任务配对结果，不含目标坐标。
- 服务器逐步轨迹：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/v2_top1_public_binding_s42_20261107/pilot128_fidelity_head/expanded/rows/`。
- Checkpoint SHA256：`e58c0fcb79603413c499c399aba8597c6b95d4b1b205f8e9d929dfed213568df`。

该结论仍未满足“完整 Val-Seen 闭环 SR>55%”目标；不把128条结果外推为全量成绩。
