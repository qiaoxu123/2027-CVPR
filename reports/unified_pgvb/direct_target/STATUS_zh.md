# 阶段状态

完成：573条合格TRAIN查询数据审计、493/80地图隔离划分、真实RGB特征、GPU训练与断点恢复、5轮925步联合训练；32条Pilot及全部56条合格Val-Seen的168次真实配对闭环。三组均为同一训练后Checkpoint的消融，非历史Geometry闭环复现。

结果与限制见REPORT_zh.md。该实验未证明新增Relation/RGB优于Geometry；不依据Val重训或调参，不访问Test，不改Stable Goal或Stop。BERT/SigLIP2冻结，不能称全骨干参数训练。

前一轮未提交的历史记忆诊断仍保留在独立目录；不将其统计与本轮结果合并。
