# 第二轮关系视觉残差实验

## 假设与实现

平均消息可能稀释关键参照；纯概率质量损失不保证 Top-1 排序。本轮增加语言/几何条件的关系注意力、空间分散锚点与可观测外观困难负样本排序。

命名地标参照：冻结 BERT 完整 token 的 masked mean，经可训练 query 投影，对地标名称 embedding 的 key 做注意力；结合公开 candidate-to-contour 边界距离，对已有观测候选打分。参照候选仍须有真实 RGB/历史支持；不会凭地图几何创造视觉证据。原始图文交互、候选几何上下文、相对方向/距离共同用于消息注意力。

地图关联不是视觉实例识别；没有人工“目标—辅助物—关系”标注。本轮不声称复现 AerialVG 关系监督。BERT token masked mean 也尚未显式解析关系短语。

## 对照与训练（查看结果前固定）

- `visual_rank_20261101`：普通视觉残差 + global rank 0.25 + observed hard rank 0.5。
- `relation_attention_rank_paired_20261101`：相同损失、batch16、seed0、lr2e-4、每轮128训练batch、3轮；增加关系注意力。
- 两者均用 native RGB/text cosine 替换 ego-distance 输入；冻结相同主干/控制器。
- 每轮 shard shuffle 后取2,048训练状态，完整4,372个Train Dev状态选 checkpoint；不以 Test/Unseen 调参。
- 完整 Dev real/zero/shuffle，关系版额外 no_relation。同损失普通版用于分离目标函数与关系模块的作用。

## 困难子集与判据

- observed_target：20m合法目标区域内存在真实有效观测。
- observed_ambiguous：上述条件且合法/非法已观测候选的冻结静态最高分差绝对值≤1。
- relation_word_observed：有观测目标且指令出现方向、between、near等关键词。这是关键词代理，不是人工关系标注；须报告样本数量。

主要指标为20m R@1、误差及配对episode聚类bootstrap，另报困难子集样本数与指标。若完整Dev关系版对同损失普通版与no_relation没有独立收益，不能把损失改变的收益归给关系模块。所有结果是离线前缀观测状态，不能当作闭环导航 SR。
