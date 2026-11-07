# 第三轮：训练时纠错与保真

依据来自TRAIN：普通CE及8m/20m组合Loss下降，但继续训练破坏开发集Top1；第一轮Birmingham确认净−8，没有扩大。第二轮公开名称修复已冻结并完成一次Val测量，Top1=48.26%，目标仍未达到。

解析器及阈值不再修改；在相同公开绑定下比较head_ce、head_fidelity、core_fidelity三组，seed42、5轮、batch32、FP32、关闭dropout使特征与推理一致。Head LR1e-4、Core LR1e-5；冻结BERT/视觉/Controller。

保真Loss只在TRAIN使用GT评价源模型：源预测正确时蒸馏原Belief并保留其排名；源预测错误时学习20m区域概率与正确区域排名，辅以弱蒸馏。无20m正网格时不制造假的区域正例。GT不进入模型forward或任何推理门控。模型架构不变，所有原权重保留。

只用Cambridge拟合11133/开发3126选择方案；不再次把已消费的Birmingham确认组作为独立确认，也不比较多组Val。开发至少+1个百分点且NE不退化，再考虑固定方案全TRAIN重训及冻结后的新学习模型最终评测。

三组各3次GPU更新Smoke通过，非零梯度，且Head组实际变化仅为评分Head及Gain，Geometry特征层未变化。首次Smoke缺少collate导入，尚未更新权重时失败；原日志保留，修复后新目录重跑。已对本轮所有新增脚本运行pyflakes，未定义名称/未使用导入均清零，补充20项相关CPU测试通过。

正式训练由tmux `v2_top1_fidelity_train_20261107`持久运行，服务器输出`GraphNav/runs/v2_top1_public_binding_s42_20261107/fidelity_adaptation/`，支持恢复。当前未宣称新训练模型达到50%。
