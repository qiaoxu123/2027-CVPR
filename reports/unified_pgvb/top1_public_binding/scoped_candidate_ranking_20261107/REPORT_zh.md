# 同一 Public Binding 下的候选评分与保守重排序验证

日期：2026-11-07。目标是排除前一轮评分头训练/推理的公开地标绑定不一致后，再判断是否值得进入 Val-Seen Pilot。

## 数据一致性修复

旧 candidate-head 的特征缓存 `public_sha256=2603f240…`，本轮闭环使用的 scoped TRAIN/Val 查询绑定为 `b8cdb6c5…`。旧评分头曾在后者上运行，因此此前128条候选头 Pilot 是跨绑定分布应用，不能视为公平的同绑定重排序结论。原结果保留，定义为绑定分布不匹配诊断。

本轮在 `scoped_data` 重新用冻结 V2 Geometry 生成完整21,878条公开输入特征，缓存哈希绑定当前 scoped public query；训练集、development、confirmation按物理目标组隔离，分别为11,133、3,126、7,619条，三组物理目标无交叉。GT只在训练Loss或前向后的TRAIN评分中使用，不进入查询、特征生成和模型输入。

## TRAIN 结果

| 方法 | Cambridge development R@20 | NE | 对slot-0纠正/破坏 |
|---|---:|---:|---:|
| Geometry slot-0 | 69.74% (2180/3126) | 22.266m | — |
| Warm-start preserving ranker最佳 | 69.67% | 22.225m | 37/39 |
| Scratch preserving ranker最佳 | 69.77% (2181/3126) | 22.123m | 45/44 |
| Scratch ranker + development置信度门控 | 69.87% (2184/3126) | 21.948m | 6/2（只切换20条） |

只看 development 会得到一个微小正向门控阈值：替代候选与slot-0分数差必须大于0.6139。该阈值随后冻结。

## Birmingham TRAIN confirmation（冻结阈值一次确认）

| 方法 | R@20 | NE | 切换数 | 纠正/破坏 | 净变化 |
|---|---:|---:|---:|---:|---:|
| Geometry slot-0 | 66.94% (5100/7619) | 24.327m | 0 | 0/0 | 0 |
| 总是按scratch ranker选候选 | 62.07% | 27.441m | 7619 | 711/1082 | -371 |
| 冻结分数门控 | 66.91% (5098/7619) | 24.837m | 105 | 19/21 | -2 |

门控从 development 的+4净纠正变为 confirmation 的-2；NE 恶化0.51m。故不满足独立 TRAIN 确认，**没有用该方法跑新的 Val-Seen 128 Pilot，更没有跑全量**。

## 判断

同绑定特征修复后，结论仍是否定的：候选分数头的信心差不能稳定区分正确替代候选；即使只在 Cambridge development 选择的少数高 margin 任务上切换，Birmingham confirmation 也略有净破坏。候选输出“进入20m”仍是位置代理，不是物理实例识别正确标签。

因此本轮不推荐把candidate-head核心化。当前完整 Val-Seen SR仍为48.54%，没有新导航成绩。后续更应获取真实 RGB 实例监督或设计具有真实可观测证据的跨模态目标匹配；不能继续在同一个排序分数上调门槛。新方案需先在TRAIN物理目标分组上通过开发与确认，再按既定128条SR≥70%门槛决定是否全量评测。

## 复现文件

- 训练脚本：`scripts/train_v2_candidate_controller.py`、`scripts/train_v2_preserving_candidate_ranker.py`。
- TRAIN margin分析脚本：`scripts/analyze_candidate_rank_margin_train.py`。门槛仅由Cambridge development选择，然后在Birmingham confirmation一次评分。
- `rank_margin_summary.json`、`rank_margin_rows.json`：逐任务开发和确认记录，无GT坐标。
- `preserving_ranker_training_summary.json`、`candidate_controller_training_summary.json`：完整训练曲线与权重SHA。
- 服务器缓存：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/v2_top1_public_binding_s42_20261107/scoped_candidate_features/`。
- 服务器Checkpoint：`.../scoped_preserving_ranker_s42/{warmstart,scratch}/best.pt`。权重未提交Git。
- scoped查询SHA：`b8cdb6c57e36f872ef493eb3c4369646f3af0ef21566ad410a445190aee09197`。

指标是TRAIN定位适配诊断，且源V2曾在官方TRAIN上预训练；不代表未见数据泛化或导航SR。
