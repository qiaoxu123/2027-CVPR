# 视觉分支关系残差验证（2026-11-01）

## 实现

`CalibratedVisualResidual(relation=True)` 在语言条件的当前/历史视觉向量上，向每个候选加入最多 16 个已观测高静态分候选的消息。消息包含视觉差向量、归一化地图 XY 位移及距离；排除自身边与无有效质量/特征的参照候选。原残差头仍以有界、中心化、置信度门控方式修改 logits。未观测区域 delta=0，末层零初始化。关系模块参数参与训练，未引入 GT 或未来帧。

这是候选级关系特征原型；没有实例身份或关系标注，也未复现 AerialVG 的框级监督。最多 16 个锚点可能遗漏重要参照物，是后续需要验证的限制。

## 缓存修复

旧默认 `GraphNav/data/graphnav_stage1_frames` 缺失。实际完整缓存为 `GraphNav/data/rebuilt_stage1_20261025/frames`，对应 BERT 缓存位于同级 `bert_tokens.pt`；已增加显式 `--cache` / `--bert-cache` 参数。重建验证记录报告 87,512 状态、21,878 episodes，并保留 Train Dev 的整 episode 划分。

## 验证设计

- 普通残差 `visual_pilot_20261101` 与关系残差 `relation_pilot_20261101`，同 seed=0、batch=16、128 batches/epoch、3 epochs。
- 每轮训练和 Dev 分别最多 2,048 状态；为固定前缀短训练诊断（smoke），不代表完整 Train/Dev 分布。
- 训练后载入最佳 checkpoint，执行 real / zero / shuffle；关系版本额外执行 no_relation。仅 Train 内 Dev 选 checkpoint，不访问 Unseen/Test。
- 原主干 SHA 在训练前后比较；smoke checkpoint 继续被闭环正式运行入口拒绝。

## 软件检查

`tests/test_calibrated_residual.py` 六项通过：包括零初始化、无效证据屏蔽、反向传播、checkpoint 严格加载及其它候选视觉信息传递。

扩展运行 `tests/test_full_resolution_controls.py` 发现一项失败：该历史测试导入共享 `GraphNav/scripts/evaluate_closedloop.py`，其中 `configure_stage1_controls` 不支持测试要求的独立 resolution 模式。本轮未修改该共享评测入口；因此不能声称全测试套件通过。

资源探测脚本未识别 GPU；实际 CUDA 训练成功，nvidia-smi 显示训练进程在 GPU 上运行。以运行日志为准。

## 短训练结果与复现

两版均以第 1 轮 Dev R@1 最优（65.4785%）选出权重。关系版 3 轮 Dev R@1 为 65.4785%、64.7949%、64.6973%；普通版为 65.4785%、64.6973%、64.7461%。后续轮次概率质量损失下降而 R@1 退化，说明仅优化目标区域概率质量不足以保证 Top-1 排序。

关系层四组参数在真实训练 checkpoint 的 Adam `exp_avg` 均非零，详见 `relation_pilot_20261101/relation_optimizer_evidence.json`。两组训练的冻结主干 SHA 检查均通过。

```sh
# 在本实验目录，使用已有 graphnav conda 环境
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 /home/20261024_1/miniconda3/envs/graphnav/bin/python -m scripts.train_calibrated_visual_residual --relation --smoke --max-batches 128 --epochs 3 --workers 2 --batch-size 16 --output <new_relation_output>
# 去掉 --relation 即为普通视觉残差对照；输出目录必须不存在。
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 /home/20261024_1/miniconda3/envs/graphnav/bin/python -m scripts.train_calibrated_visual_residual --evaluate-checkpoint relation_pilot_20261101/best.pt --smoke --workers 2 --batch-size 16 --output <new_eval_output>
```

训练样本随每轮 shard shuffle 变化，每轮各取 128 batches；Dev 前缀固定。完整 Dev 验证不设 `--max-batches`。所有本轮权重仍属 smoke，未运行闭环导航评测。

## 完整 Dev 结果

|模型/干预|状态数|R@1 (20m)|平均误差 m|
|---|---:|---:|---:|
|relation/real|4372|67.5892%|20.7144|
|relation/zero|4372|67.5206%|20.7206|
|relation/shuffle|4372|67.5663%|20.7105|
|relation/no_relation|4372|67.5892%|20.7195|
|visual/real|4372|67.5892%|20.7122|
|visual/zero|4372|67.5206%|20.7206|
|visual/shuffle|4372|67.5663%|20.7103|

配对结果（按 episode 聚类、5,000 次 bootstrap；这些是重复前缀观测状态，非闭环 SR）：

- relation_vs_visual: +0.0000pp，95%CI [0.0, 0.0]，帮助 0 / 损害 0 状态。
- relation_vs_zero: +0.0686pp，95%CI [-0.29734675205855443, 0.434583714547118]，帮助 14 / 损害 11 状态。
- relation_vs_no_relation: +0.0000pp，95%CI [0.0, 0.0]，帮助 0 / 损害 0 状态。
- relation_vs_shuffle: +0.0229pp，95%CI [0.0, 0.0686184812442818]，帮助 1 / 损害 0 状态。

关系版删除关系模块后 R@1 没有变化，真实 RGB 对打乱 RGB 的差异也很小。本轮证明实现可训练，但不支持关系模块提升导航或定位准确率的结论。建议下一轮优先加入关系困难样本/可靠参照监督与排序目标，再扩大训练；不要直接并入正式主结果。
