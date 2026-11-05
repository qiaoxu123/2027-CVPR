# PR17 正式训练阶段报告

完整Revised TRAIN：21878条，转换0跳过，目标坐标无越界/非有限值，91条无公开参考仍参与训练。seed42，BERT和SigLIP2冻结。

Geometry+Relation联合训练5轮，每轮21878条，总13675更新。记录的训练时间1166.07秒（约19.43分钟）；Loss 3.452412→3.424544。峰值PyTorch GPU allocated 2.154GiB，单步时间不包含Dataset取样等待，完整时间以轮次Wall Time为准。

真实RGB缓存：249帧，205条fit指令；PGVB三轮共747次样本参与优化，11.70秒；Loss 3.547072→3.528090。Geometry/Relation保持逐张量不变，PGVB19个权重张量实际改变。不宣称全量视觉训练或官方AerialVG模型预训练。

364项CPU测试通过。两条TRAIN真实闭环接口Smoke通过，但不是SR评测。原始模型SHA未改变。

全Val-Seen2470任务×三组匹配消融及额外原Geometry完整基线正在运行；模型使用最后固定轮次Checkpoint，不根据Val分数调整或选择。当前不提供尚未完成的SR结论。

Checkpoint路径与SHA见checkpoint_SHA256.json；训练曲线见training_curves.png；服务器原始日志保持在runs/unified_full_relation_s42_20261105。
