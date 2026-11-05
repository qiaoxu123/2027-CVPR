# 保留目标＋特征＋地标＋关系：完整TRAIN训练及2470条导航


全部2470/2470条Val-Seen真实闭环。Pilot后未改变模型或配置，不使用Val选择Checkpoint或调参，未运行Test。未修改Stop/Goal/控制器。

## 输入与训练

全部21878条TRAIN文本本身不含可删除的开头套话，本轮实际完整保留原文，数据SHA与上一轮public-bound full_text完全相同。没有声称完成LLM重构、可靠四槽语义解析或新增外观网络；目标外观及实例关系仍由原BERT Token通道输入。

沿用原架构、原V2初始化、seed42、5轮、batch8、Geometry1e-5/Relation1e-4、CE+0.05KL、FP32。训练Geometry和Relation，PGVB/BERT/控制器冻结；静态文本消融RGB关闭，不能解释为验证了视觉识别。13675次更新，耗时1115.46秒。PGVB等未训练模块逐张量核对与初始化完全相同。

与上一轮精简模型使用相同公开Polygon绑定和训练设置。本轮是补齐完整内容在新绑定下的训练对照；旧全文Checkpoint在旧绑定下训练，因此与它的差异也包含重新绑定后的训练适配。

## 固定2470条配对闭环

包含全部2470条Val-Seen，不按候选可用性、成功或GT过滤。新模型真实闭环；两个旧模型复用之前完整评测中完全相同的2470条轨迹，逐条核对Episode、起点、公共Polygon及世界范围。无候选和失败均入分母。指标为Episode平均。

|方法|SR|OSR|SPL|NE(m)|nDTW|步数|
|---|---:|---:|---:|---:|---:|---:|
|old_full|45.75%|53.28%|0.4440|37.960|0.2888|6.84|
|compact|34.09%|41.54%|0.3297|41.420|0.2766|6.89|
|target_features|47.37%|54.01%|0.4598|34.226|0.2948|6.83|

相对旧完整指令：{'fixes': 97, 'damage': 57, 'net': 40}；相对精简：{'fixes': 482, 'damage': 154, 'net': 328}。物理目标组bootstrap95% SR差值：{'old_full': [0.003654006270490892, 0.029280283712641977], 'compact': [0.10644070133679001, 0.1611780513176144]}。一个训练种子；本次全量Seen结果不能替代Unseen泛化。不能作为多种子泛化结论。

导航失败归因（GT仅事后使用）：{'old_full': {'failed_with_wrong_final_goal': 1340, 'OSR_success_SR_failed': 186, 'correct_policy_stop': 1130, 'goal_jumps_over20m': 0, 'policy_stop': 2470, 'failed_with_correct_final_goal': 0}, 'compact': {'failed_with_wrong_final_goal': 1628, 'OSR_success_SR_failed': 184, 'correct_policy_stop': 842, 'goal_jumps_over20m': 0, 'policy_stop': 2470, 'failed_with_correct_final_goal': 0}, 'target_features': {'failed_with_wrong_final_goal': 1300, 'OSR_success_SR_failed': 164, 'correct_policy_stop': 1170, 'goal_jumps_over20m': 0, 'policy_stop': 2470, 'failed_with_correct_final_goal': 0}}。

新增推理错误：0。新模型平均单步推理24.409ms；旧模型为历史缓存耗时，不是同一时段速度对照。

Checkpoint: /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v2_target_features_s42_20261106/train/last.pt
SHA256: 9ac51ee253eed46acbf66e6afb47888857b70258f5fcd195721723d5112dc387

原始训练日志、Checkpoint和逐步Belief/轨迹保留服务器/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v2_target_features_s42_20261106；GitHub包含配对CSV、固定任务清单、曲线及典型成功/失败热图轨迹。世界目标20m指标不证明真实实例匹配。公开地图同名Polygon统计见上级实验目录homonymous_polygons.json；道路分段不能直接解释为不同实例。

## 结论与验证

相对旧全文SR增加1.62个百分点，净成功+40；NE降低3.73m。相对精简组SR增加13.28个百分点，净成功+328，说明保留目标描述和实例约束优于本轮删减方案；但无法将相对旧全文的收益单独归因于语义理解，因为重新绑定后的训练适配也改变了条件。

按449个物理目标分组bootstrap，较旧全文的SR差值95%区间为[+0.37,+2.93]个百分点；仍是单训练种子，不宣称多种子或Unseen提升。

1300条失败均伴随最终选定Goal错误；这不是所有动作/Stop均正确的证明。164条OSR成功而SR失败，可能存在更早停止的理论空间，但没有根据GT改变停止机制。优先保留完整目标信息；本次没有验证RGB带来的额外收益，不应把该结果称为新视觉识别能力。

395项CPU测试通过、真实GPU训练和全量闭环完成、冻结模块逐张量一致。全量评测实际耗时597.82秒，峰值GPU分配3.745GiB。独立adversarial-review已调用，但服务workspace routing discovery timed out，未计为审查通过。全部原Checkpoint与旧报告保留，未访问Test，未合并main。
