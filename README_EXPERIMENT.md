# 冻结模型的全局地标上下文验证（2026-10-16）

在原冻结 GraphNav-V1 C Epoch1 上进行 28×28、stop=0.3 的 Val-Seen 输入干预，不训练任何网络，不覆盖正式结果。此 worktree 从视觉残差分支明确 SHA `0e59f7d2572294f1c4c76a37c7d62b5f23991e15` 创建；继承的旧视觉残差报告不是本轮新结果。

对照组 references 使用原引用地标；all_named 在几何注意力中追加 SBF 全局轮廓通道所用的具名地标（每个精确名称取最大多边形），并保留原引用碎片。原引用先验、规则、语言、视觉、历史和控制器保持一致；不按目标、无人机位置或可见性筛选。all_named_masked 只用于扩展输入的屏蔽校验。

这是全局具名地标节点输入的冻结鲁棒性测试。模型没有新增全局轮廓栅格编码器，没有加入未经训练的引用标记权重，因此不能当作新增网络训练后的结论。

运行 `python -u -m scripts.run_landmark_context`。进程互斥；已完成任务跳过，未完成 episode 通过 evaluator 的 --resume 恢复。先运行4条 references/masked/all_named 实现检查，再运行同一 Seen128 的 references/all_named，最后运行两组完整 Val-Seen2470。失败保留日志，不自动跳过失败样本。所有输出位于 landmark_context_results/。

查看 frozen_plan.json、queue.json、progress.json、comparison.csv 和 paired_comparison.json。配对汇总仅在对应两组完整结束后生成，报告 SR/SPL/NE、初始静态定位、帮助/损害数量和配对 bootstrap 区间。Test/Unseen 不参与本轮选择。
