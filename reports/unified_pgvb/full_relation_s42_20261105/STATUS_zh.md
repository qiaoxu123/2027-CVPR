# PR17全量TRAIN与全Val-Seen：阶段状态

基准PR17：4e37f70；执行docs/UNIFIED_FULL_RELATION_TRAIN.md。

已完成：Revised TRAIN 21878条全部转换，0跳过；不使用573条严格清单作为训练资格。361项CPU测试通过（新增padding等价测试已包含）。真实GPU 2批Smoke：Geometry/Relation梯度非零，第二次更新出现非零关系残差。

正在运行：tmux unified_relation_core_20261105，seed42，5轮Geometry+Relation，FP32 Belief。修复轮内每200更新保存、确定性顺序恢复、BERT特征缓存及无效padding计算。不改任何Stop/Goal阈值。

待执行：已有TRAIN真实RGB缓存的3轮PGVB训练；冻结后全部2470条Val-Seen三组闭环，含错误/无参考任务。不会用Val指标选择Checkpoint或参数。当前没有完整SR结果，不宣称性能改善。

保持BERT和SigLIP2冻结，遵循PR17文档；Phase1训练Geometry与Relation，Phase2仅训练PGVB。不能称全视觉骨干端到端训练。实际adversarial-review工具已调用，但workspace routing discovery失败，未取得独立审查通过。

服务器日志：/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105/。

恢复验证：从真实训练的第1000步Checkpoint重载模型/优化器，模型权重逐张量一致；本轮已见8000条+剩余条数恰等于21878。RGB阶段只有249个真实当前帧（205条fit指令）具有与运行时一致的局部Anchor支持，另1067条被明确记录为不可用/歧义/画外；不宣称视觉全样本训练。

全Val-Seen计划三组同一训练后Geometry配对消融，额外保留原始Geometry完整基线。Geometry、Relation、RGB三组均为2470完整任务，外加原始Geometry2470；不根据结果选参。
