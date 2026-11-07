# 保真微调 Geometry 的 Birmingham TRAIN Confirmation 检查

日期：2026-11-07。比较原 Unified V2 Geometry 与 TRAIN 保真微调 Checkpoint，数据仅取此前未参与本轮头部选择的 Birmingham confirmation：7,619条指令，1,386个物理目标组。fit、development、confirmation 组间无物理目标组重叠。公开输入使用中性目标坐标占位；目标坐标只在前向完成后计算 R@20 和 NE。

| 指标 | 原 V2 Geometry | 保真微调 | 差值 |
|---|---:|---:|---:|
| R@20 | 67.082% (5,112/7,619) | 67.200% (5,121/7,619) | +0.118 pp (+9条) |
| NE | 24.292 m | 25.179 m | +0.886 m（更差） |
| 配对纠正 | — | 179 | — |
| 配对破坏 | — | 170 | 净+9 |

配对R@20差值的近似95%区间为[-0.36,+0.60]个百分点，包含零。该区间按任务独立近似计算，未处理同物理目标分组相关性，因此只作描述性参考。NE恶化与R@20的小幅上升同时出现。

地图分布不均：block12净变化-2.30 pp（15纠正、33破坏）；block1 +0.57 pp；block10 +2.09 pp；block3 -0.33 pp；block4 +0.22 pp；block7 +0.29 pp；block9 +0.66 pp。样本量在地图间差异很大，不据此选择地图规则。

## 结论

微调头在 Cambridge development 的R@20曾从69.39%到70.06%，但在 Birmingham confirmation 仅净增9/7619，NE明显变差；因此跨城定位收益很弱，不能支持继续扩大该训练方案。基模型本身已经在官方TRAIN上预训练，confirmation只能检验相对适配的遗忘/迁移，不是未见训练数据的泛化证明。

保真 Checkpoint 的 Val-Seen 128条Pilot为56.25% SR，低于用户设定的70%全量筛选门槛；完整闭环评测没有运行。短期最应优先解决的是关系/目标监督的跨场景有效性和视觉实例证据质量，而不是继续重复候选分数头或全局路径控制消融。当前完整 Val-Seen SR仍为48.54%，目标未达成。

## 复现

- `summary.json`：分组、两个 Checkpoint SHA、计时、精度和脚本SHA。
- `paired_rows.csv`：7,619条逐任务结果，不包含GT坐标。
- `baseline_rows.json`、`adapted_rows.json`：轻量逐样本评分结果。
- 运行代码：`scripts/evaluate_v2_geometry_train_confirmation.py`。
- 预训练基线SHA256：`9ac51ee253eed46acbf66e6afb47888857b70258f5fcd195721723d5112dc387`。
- 保真微调Checkpoint SHA256：`e58c0fcb79603413c499c399aba8597c6b95d4b1b205f8e9d929dfed213568df`。
