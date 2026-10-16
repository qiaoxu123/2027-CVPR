# 冻结 GraphNav-V1 的 Stop 到达门控对照（2026-10-16）

本 worktree 从 `0e59f7d2572294f1c4c76a37c7d62b5f23991e15` 创建，正式模型固定 C Epoch1、原引用地标、28×28、Stop 校正分数阈值0.3、原高度门控与控制器。不训练，不修改主结果。

唯一干预是无人机到当前预测目标的 XY 距离容差：0.001/1/5/10/20米；-1仅取消该距离门控，仍要求Stop分数及高度条件通过。策略不读取真实目标。每组重新运行完整闭环，不从已截断轨迹外推。先六组相同均匀Seen128，再六组完整Val-Seen2470，不使用Unseen/Test选择参数。

入口 `python -u -m scripts.run_stop_arrival_sweep`；输出在 stop_guard_results/。进程互斥；已完成项跳过，未完成输出使用--resume。frozen_plan.json记录冻结配置，queue.json/progress.json提供状态，comparison.csv提供SR/SPL/NE/步数/错误Stop率，paired_comparison.json提供相对1毫米基线的帮助/损害样本与配对bootstrap区间。没有自动替换正式配置。

本轮只测试推理门控，不能证明任何训练重设计的效果。继承的视觉残差元数据属于祖先实验，不能作为本轮结果。
