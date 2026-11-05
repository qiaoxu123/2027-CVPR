# 保留目标＋特征＋地标＋关系：完整TRAIN训练及128条导航Pilot

仅诊断，不是2470条正式全量成绩。未运行新全量Val或Test，未修改Stop/Goal/控制器。

## 输入与训练

全部21878条TRAIN文本本身不含可删除的开头套话，本轮实际完整保留原文，数据SHA与上一轮public-bound full_text完全相同。没有声称完成LLM重构、可靠四槽语义解析或新增外观网络；目标外观及实例关系仍由原BERT Token通道输入。

沿用原架构、原V2初始化、seed42、5轮、batch8、Geometry1e-5/Relation1e-4、CE+0.05KL、FP32。训练Geometry和Relation，PGVB/BERT/控制器冻结；静态文本消融RGB关闭，不能解释为验证了视觉识别。13675次更新，耗时1115.46秒。PGVB等未训练模块逐张量核对与初始化完全相同。

与上一轮精简模型使用相同公开Polygon绑定和训练设置。本轮是补齐完整内容在新绑定下的训练对照；旧全文Checkpoint在旧绑定下训练，因此与它的差异也包含重新绑定后的训练适配。

## 固定128条配对闭环

从2470条Val-Seen按既有evenly_spaced_indices确定128条，不按成功或GT选择。新模型真实闭环；两个旧模型复用之前完整评测中完全相同的128条轨迹，逐条核对Episode、起点、公共Polygon及世界范围。无候选和失败均入分母。指标为Episode平均。

|方法|SR|OSR|SPL|NE(m)|nDTW|步数|
|---|---:|---:|---:|---:|---:|---:|
|old_full|43.75%|49.22%|0.4282|35.953|0.2895|7.09|
|compact|32.81%|39.84%|0.3233|40.651|0.2703|7.00|
|target_features|42.97%|49.22%|0.4188|32.990|0.2923|7.05|

相对旧完整指令：{'fixes': 3, 'damage': 4, 'net': -1}；相对精简：{'fixes': 24, 'damage': 11, 'net': 13}。物理目标组bootstrap95% SR差值：{'old_full': [-0.047244094488188976, 0.03125], 'compact': [0.007874015748031496, 0.18604651162790697]}。一个训练种子、小样本Pilot，不能作为全量SR或多种子泛化结论。

新增推理错误：0。新模型平均单步推理26.032ms；旧模型为历史缓存耗时，不是同一时段速度对照。

Checkpoint: /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v2_target_features_s42_20261106/train/last.pt
SHA256: 9ac51ee253eed46acbf66e6afb47888857b70258f5fcd195721723d5112dc387

原始训练日志、Checkpoint和逐步Belief/轨迹保留服务器/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v2_target_features_s42_20261106；GitHub包含配对CSV、固定任务清单、曲线及典型成功/失败热图轨迹。世界目标20m指标不证明真实实例匹配。公开地图同名Polygon统计另见homonymous_polygons.json；道路分段不能直接解释为不同实例。

## 测试、复现及执行范围

395项CPU测试通过（1项既有rasterio弃用警告），训练GPU Smoke通过，Geometry/Relation梯度非零且Loss有限。原评测器默认只接受全量；本轮增加必须显式声明的--diagnostic-pilot，任务128/128完整执行、总清单2470条，不能冒充全量。Stop阈值0.3、到达守卫1mm、Horizon20和动作策略不变。

训练配置及真实运行命令见training_manifest.json、train.sh，导航命令见pilot128.sh；按原训练脚本添加--resume可从last.pt恢复，脚本每200步保存优化器与随机状态。不要直接覆盖已有结果。GPU为RTX PRO6000 Blackwell，实际设备状态见nvidia_smi.csv；通用资源脚本因PATH中没有nvidia-smi误报无GPU，不作为设备依据。

当前判断：相较丢失目标描述的精简输入，完整内容恢复13次净成功（+10.16个百分点）；相较旧完整指令少1次成功（−0.78个百分点），NE下降约2.96m。尚不能证明超过旧全文模型，不继续根据这128条调参或自动扩大评测。

后续执行状态：用户查看Pilot后授权继续全量2470条Val-Seen，使用同一Checkpoint、不调参。完整结果将独立保存在[full_val_seen/REPORT_zh.md](full_val_seen/REPORT_zh.md)，本文件保留原128条结果，禁止混用分母。保持原BERT max_length96；原始11条TRAIN长指令仍受相同截断约束，未宣称所有长指令Token均完整进入模型。

独立adversarial-review已实际调用，但服务返回workspace routing discovery timed out，未取得有效审查结果，不计为审查通过。
