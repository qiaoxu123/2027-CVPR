# 固定Geometry目标下的直达航点控制 Pilot

日期：2026-11-07。对同一 Val-Seen 前128条任务运行配对 Pilot；只将 Stage1 预测的首个Geometry Goal航点改为受限直达航点，Goal选择、Stage1 权重、Stop阈值0.3、到达守卫、Horizon、动作步长与20m成功判据保持不变。没有用GT控制动作或Stop。

| 指标 | 原 guarded-blend | guarded-direct | 差值 |
|---|---:|---:|---:|
| SR | 53.91% (69/128) | 53.91% (69/128) | 0 |
| OSR | 57.81% | 58.59% | +0.78 pp |
| SPL | 52.75% | 52.96% | +0.21 pp |
| NE | 24.98 m | 24.98 m | 约0 |
| nDTW | 0.2934 | 0.2944 | +0.0009 |
| 平均步数 | 7.39 | 5.03 | -2.36 |
| 平均推理时间 | 5.51 ms | 6.38 ms | +0.86 ms |

逐任务变化：纠正0条、破坏0条，SR配对完全相同。直达航点减少了步数，但未增加成功数。它不能证明控制器从来不是瓶颈，但当前证据不支持仅改变路径跟随方式即可达到目标。

按用户设定的筛选门槛，SR未达到70%，**未运行全量评测**。本结果只代表固定的128条Pilot。

## 可复现信息

- `manifest.json`：数据、Checkpoint、代码SHA、协议。
- `metrics.json`：聚合结果。
- `paired_episode_outcomes.csv`：逐条配对结果。
- 完整逐步记录保存在服务器：`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/v2_top1_public_binding_s42_20261107/pilot128_guarded_direct/expanded/rows/`。
- `tests/test_a3_trajectory_runtime.py`：3项通过。
