# GraphNav 工程命名与诊断 TODO（2026-10-13）

正式名称：GraphNav — Graph-First Aerial Vision-and-Language Navigation；主模型 GraphNav-V1。
目录 `2027-CVPR/GraphNav`、包 `graphnav`；冻结 C Epoch1 正式结果保持独立。

- [x] 核验名称、路径、checkpoint、已有修改与运行任务；统一当前工程说明。
- [x] 按 HETT 口径执行逐决策 profiling：rollout、语言、视觉、静态图、视觉残差/历史、轨迹 Stop、观测、其他；总体方差、首条与排除首条、最慢五条。
- [x] 测试真实 heatmap 网格 28→384（照片 384×384）；记录显存、延迟与导航变化，不能以插值冒充精度提升。
- [x] 验证 stop 阈值确实生效；同一 Val-Seen 子集闭环测试 0.1/0.3/0.5/0.7/0.9，比较 SR/SPL/NE、错误 Stop、终止原因和步数。
- [x] 汇总可复现命令、输入/权重 SHA 与限制，更新此 TODO。

实验均写入 `2027-CVPR/Ablations/GraphNav/diagnostics_20261013/`。
不使用 Test 调阈值。当前 HTNav 并发训练会影响耗时，结果需记录该限制。

结果：[完整诊断报告](../../Ablations/GraphNav/diagnostics_20261013/RESULTS.md)。10 项评测全部完成，相关测试 26 项通过。

结论：视觉残差在 Seen128 上没有正收益（SR -0.78pp）；真实 384 网格仅 4 条样本可行性验证，约 3.14s/决策、12.32GiB 峰值。stop 0.1/0.3 相同，提高阈值主要延长决策并增加预算/停滞终止。本次未改变冻结正式配置。HETT 整 episode CUDA Event profiling 为额外 Unseen128，性能结论受 HTNav 并发影响。
