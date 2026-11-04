# Changelog

## [unified-pgvb-v2-training-pilot] - 2026-11-04
### Features
- 新增通过UnifiedPGVB-v2实际前向/保存的PGVB视觉Head训练入口，支持BF16、seed42、优化器/RNG/帧游标恢复及同步CUDA效率日志。
- 扫描22,002条TRAIN原始起点，增量生成4,159帧/4,658查询的原生yaw、公开Polygon弱监督缓存；完成200步、额外1000步及全量可用t=0拟合池单Epoch。
- 保存独立TRAIN Dev、Geometry-only/训练前后V2、RGB置零、真实Belief图及前后各2条闭环功能Smoke。
### Design Rationale
- 仅训练唯一UnifiedPGVB内视觉Head，冻结Geometry/主干/控制器；静态Geometry无须在局部损失每步重复计算，不把迁移冒充联合训练。
- 当前合法观察与未知监督分开；全量t=0分母与原增强缓存分开报告，首次缓存及后续复训成本独立计时。
### Notes & Caveats
- 公开Polygon弱代理不是人工像素Gold；Other无视觉监督，方向、目标外观、joint融合未训练。
- 局部Loss/R20改善但Polygon概率质量降低，融合R20不变；两条固定功能Smoke前后均0/2成功，无完整SR结论，不自动进入10Epoch。
- 原main、archive和历史Checkpoint不改写；34项测试通过，真实GPU梯度、精确恢复及运行时加载已核验。
