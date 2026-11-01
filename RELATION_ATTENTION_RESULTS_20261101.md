# 第二轮关系视觉残差训练验证

协议：[RELATION_ATTENTION_PROTOCOL_20261101.md](RELATION_ATTENTION_PROTOCOL_20261101.md)。仅以 Train Dev 选权重；没有使用 Unseen/Test。

## 本轮改变

- 使用指令编码与地标名称编码的注意力、公开轮廓距离，为已观测参照候选提供关联分数。
- 用候选视觉、语言、几何上下文和相对位置计算消息权重；空间分散选择最多16个参照，替代平均聚合。
- 加入全局排序损失（0.25）及已观测视觉困难负样本排序损失（0.5）。普通视觉分支使用同样目标，区分损失改变与关系结构的效果。
- 每轮128训练batch（2,048状态）共3轮；本轮改用完整4,372个Train Dev状态选checkpoint。
- `--native-contrast` 将 ego-distance 输入换成原生图文cosine；两版一致。

## 初始化与软件验证

新增可选模块原先会消耗共享层初始化随机数，现已修正并测试共享参数初值完全相同。关系运行 `relation_attention_rank_20261101` 已停止、保留为 superseded；有效对照为 `relation_attention_rank_paired_20261101` 对 `visual_rank_20261101`。

9项相关测试通过，涵盖共享初始化一致、注意力/命名地标分支反向梯度、未观测屏蔽、合法/非法候选排序梯度、历史checkpoint加载和跨候选视觉信息传递。真实32状态smoke的前向/反向、完整性检查与控制均完成。

## 限制

- 仍没有人工目标—参照物关系标注，也没有使用GT为forward生成参照物。
- 地图轮廓关联不能声称视觉实例匹配；完整BERT token的masked mean尚未显式解析关系短语。
- 困难子集是基于观测覆盖、冻结分差和指令关键词的代理，必须结合样本数解释。
- 训练为短预算，不是完整Train收敛训练；本轮没有闭环导航评测。

## 完整 Dev 结果

|模型/干预|状态数|R@1 (20m)|平均误差 m|
|---|---:|---:|---:|
|relation/real|4372|67.7951%|20.5789|
|relation/zero|4372|67.5206%|20.7206|
|relation/shuffle|4372|67.5206%|20.6113|
|relation/no_relation|4372|67.7264%|20.5619|
|visual/real|4372|67.7951%|20.5881|
|visual/zero|4372|67.5206%|20.7206|
|visual/shuffle|4372|67.5435%|20.6135|

配对结果（按 episode 聚类、5,000 次 bootstrap；这些是重复前缀观测状态，非闭环 SR）：

- relation_vs_visual: +0.0000pp，95%CI [-0.16010978956999086, 0.16010978956999086]，帮助 4 / 损害 4 状态。
- relation_vs_zero: +0.2745pp，95%CI [-0.3202195791399817, 0.9149130832570905]，帮助 44 / 损害 32 状态。
- relation_vs_no_relation: +0.0686pp，95%CI [-0.11436413540713632, 0.29734675205855443]，帮助 8 / 损害 5 状态。
- relation_vs_shuffle: +0.2745pp，95%CI [0.04574565416285453, 0.5489478499542544]，帮助 17 / 损害 5 状态。

## Real 条件的困难子集

|模型/子集|样本数|R@1|平均误差m|
|---|---:|---:|---:|
|relation/static/observed_target|1967|68.7341%|20.8931|
|relation/static/observed_ambiguous|1583|67.5932%|19.8542|
|relation/static/relation_word_observed|1414|70.1556%|19.8814|
|relation/corrected/observed_target|1967|69.7001%|20.4894|
|relation/corrected/observed_ambiguous|1583|68.6039%|19.4442|
|relation/corrected/relation_word_observed|1414|71.2871%|19.2279|
|visual/static/observed_target|1967|68.7341%|20.8931|
|visual/static/observed_ambiguous|1583|67.5932%|19.8542|
|visual/static/relation_word_observed|1414|70.1556%|19.8814|
|visual/corrected/observed_target|1967|69.7001%|20.5099|
|visual/corrected/observed_ambiguous|1583|68.4776%|19.4845|
|visual/corrected/relation_word_observed|1414|71.2871%|19.2606|

以上为离线定位诊断，不能表述为闭环导航收益。关系模块的独立作用须同时对照同损失普通版与no_relation；关系词与静态分差仅为困难样本代理。

## 判断与统计边界

- 两版整体R@1均为67.7951%；关系版对普通版差值0.000pp，配对区间[-0.160, +0.160]pp。关系结构的独立准确率增益未得到支持。
- 删除关系模块后R@1下降0.0686pp，区间[-0.114, +0.297]pp，仍跨零；困难分差子集关系版仅比普通版多命中2个状态。
- 真实RGB优于打乱RGB 0.2745pp，区间[+0.046, +0.549]pp：相比第一轮原型，当前配置出现依赖视觉空间对应的信号。该变化同时涉及损失函数、native-contrast及结构，不能独归因于关系注意力。
- 整体相对冻结静态/zero为+0.2745pp，区间[-0.320, +0.915]pp，尚未明确优于静态基线。
- 同一完整Dev参与checkpoint选择，这些bootstrap区间仅是选择集上的探索性描述，不是独立泛化检验；每episode四个前缀状态，已按episode聚类。
- 依照进入主线门槛，本轮不开展以“关系提升”为前提的大规模闭环实验。后续若继续，优先检验可靠目标—参照物绑定和关系短语监督，并用独立验证划分及多seed确认。

两组训练均完成冻结主干SHA完整性检查。已保留权重、逐状态预测、完整Dev干预结果、协议及最终源码快照；未改写既有正式主结果。
