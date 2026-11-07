# Top-1超过50%：阶段进展

**当前未完成目标。** 同一冻结训练后V2 Geometry、FP32单条推理，完整HETT Refined Val-Seen2470条：公开名称提取修复使Top1从46.96%升至48.26%，纠正41、破坏9、净+32。NE从35.1966m降至33.6669m；空间独立Top5从77.85%升至79.43%。

此前47.09%来自BF16批量推理；本轮配对基线46.96%来自FP32单条推理，不混合两种精度口径。两组候选网格、20m标准、Checkpoint及全文语言相同，仅公开名称绑定/Alias改动。GT只在forward后评分，未通过GT筛地标、候选或视点。没有运行新导航或Test，也不宣称实例识别准确率。

[第一轮训练负结果](phase1/REPORT_zh.md)：三个Loss方案各5轮，开发最优模型独立TRAIN适配确认净−8，因此没有扩大或拿它跑Val。[第二轮名称解析](phase2/REPORT_zh.md)：只在Cambridge TRAIN开发组修正语义作用域，保守版在Val评分前冻结；完整指标与逐任务对照已保存。没有据Val再改解析阈值。

[第三轮保真训练](phase3/PLAN_zh.md)正在运行：名称解析器保持冻结，比较仅评分Head/Geometry核心，训练时对源模型正确预测保真、对错误预测纠正。GT只决定TRAIN Loss，不进入推理。只用TRAIN开发选择，已消费的Birmingham确认组不重用为独立确认。

分支experiment/v2-top1-public-binding-20261107。原V2/V3/控制器权重未覆盖，不改main、Stop、Horizon或命中标准。服务与此前配准结果未动。真实权重仍在服务器，Git只保存代码与轻量结果。
