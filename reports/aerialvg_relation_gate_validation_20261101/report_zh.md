# AerialVG Relation/Gate 训练验证（2026-11-01）

结论：固定 seed=0、4 轮 Relation + 3 轮 Gate **未带来超越原 Native Residual 的定位提升**。关系特征确实影响排序，但有效监督稀少、Attention 退化为单参考；Gate 的预测概率未进入预设阈值范围，全部回退 A0。当前不建议替换原方案。

## 分支与范围

- 拉取并验证源分支 `research/aerialvg-relation-gated-grounding-20261101`，完整基点 `a14292c1677964e358b278c7e948f0abfecf929a`。
- 修复和报告位于独立分支 `validation/aerialvg-relation-gated-20261101`。原分支、原正式实验、冻结权重均保留。
- 仅用缓存 TRAIN 1275 状态/252 组与内部 Dev 635 状态/125 组；TRAIN 按已有 group-hash 分为 fit 1027/203 组、calibration 248/49 组，组间无重叠。
- 不训练 AerialVG 主干；冻结 Native 文件 SHA256 为 `5956313f58b62315e7f56a8f6710d9cb8fb3b600b479c916d859d1c898d5578a`。结束时校验未改变；公开参考缓存亦通过校验。A0 直接复用缓存，不因解析修复而重算。
- 每次训练在模型与 TRAIN 校准阈值冻结后才评估内部 Dev；smoke 与 full 分别评估，随后只读诊断已冻结 full 模型。轮数由任务预先指定，未按 Dev 选择配置。未运行闭环或正式 Val-Seen、Val-Unseen、Test。

## 测试及修复

原指定两文件：23 passed。补充测试先复现 2 个失败，再修复后 **29 passed**；扩展相关路径 **43 passed**（1 个上游 DeprecationWarning）。

1. 修复 `north-east` / `north east of` 被误识别为 east；仅规范化紧邻 `of` 的复合方向短语，避免重写参考名称。
2. Masked 文本用 `masked_fill` 清零，避免 `NaN * 0` 污染；活动文本、几何、Native null 非有限值直接报错。
3. 无正负候选或无唯一关系的 batch 不再执行 AdamW step，避免零监督时仍发生权重衰减/动量更新；记录真实监督行数、梯度和参数差量。
4. 分开记录状态/组数，修正旧脚本将 relation states 误称 groups 的统计。新增关闭关系/打乱候选关系几何的消融与只读诊断脚本。

检查覆盖世界 east/north 符号及平移不变性、query/reference/text Mask、单/双参考梯度、零初始化、无/歧义关系精确回退、GT 输入隔离。GT 仅用于训练 loss/训练校准标签及离线指标；修改 GT 不改变公开关系描述符。冻结 Native 无梯度。

扩展检查另发现基点已有的 `tests/test_relation_supervision.py` 在收集时引用不存在的 `legal_region`（同文件还引用旧 `RelationDataset` API）。该测试属于旧关系监督实现，本轮未恢复已移除 API、未据此修改正式实验；完整错误见 `legacy_test_collection_failure.txt`。因此不声称仓库全量测试通过。

## 相同 635 状态的 full 对比

R@1：Belief 最终峰值距 GT ≤20m；NE：该峰值的平均世界距离。候选排序正确：有效非 Null query 的 argmax 距 GT ≤20m。**20m 空间效用标签不等于真实目标实例匹配**。A0 无视觉候选排序；Gate 行展示融合前的候选排序。纠正/误改/净收益均相对 A0。

| 方法 | R@1 | NE (m) | 候选排序正确/635 | 纠正 A0 | 误改 A0 | 净收益 |
|---|---:|---:|---|---:|---:|---:|
| A0 | 68.82% | 22.2420 | — | 0 | 0 | +0 |
| 原 Native Residual | 69.92% | 21.8517 | 273/635 (42.99%) | 22 | 15 | +7 |
| Relation Residual | 69.76% | 21.8844 | 275/635 (43.31%) | 22 | 16 | +6 |
| Relation + Gate | 68.82% | 22.2420 | 275/635 (43.31%) | 0 | 0 | +0 |
| 关闭关系 + Residual | 69.92% | 21.8517 | 273/635 (42.99%) | 22 | 15 | +7 |
| 打乱关系 + Residual | 69.61% | 21.9243 | 272/635 (42.83%) | 22 | 17 | +5 |
| 关闭关系 + Gate | 68.82% | 22.2420 | 273/635 (42.99%) | 0 | 0 | +0 |
| 打乱关系 + Gate | 68.82% | 22.2420 | 272/635 (42.83%) | 0 | 0 | +0 |

候选 20m 覆盖为 **391/635 = 61.57%**。仅在覆盖状态内，Native 排序正确率为 273/391 = 69.82%，Relation 为 275/391 = 70.33%，打乱关系为 272/391 = 69.57%。关闭关系 logits 精确回退 Native；无具名关系的 560 状态也精确保持 Native logits。

打乱采用固定、无标签的有效 query 顺序反转，只打乱 query 对应的关系几何，保留 Native 特征/分数、参考种类和 Mask；不是简单置换 attention 参考槽位。Gate 使用同一权重、同一 TRAIN 阈值，消融不重训。

Smoke 1+1：Relation R@1 69.92%、NE 21.8517m，与 Native 相同；Relation+Gate 为 A0 的 68.82%、22.2420m。Smoke 数值有限、梯度/更新正常、不可用关系精确回退后，按预设执行 full。

## 有效关系与训练证据

| 集合 | 状态/组 | 有唯一具名关系：状态/组 | 同时有正负候选、可监督：状态/组 |
|---|---:|---:|---:|
| TRAIN fit | 1027 / 203 | 85 / 17 | 49 / 17 |
| TRAIN calibration | 248 / 49 | 15 / 3 | 11 / 3 |
| 内部 Dev | 635 / 125 | 75 / 15 | 41 / 15 |

- TRAIN 总计 100/1275（7.84%）状态有唯一具名关系；Dev 为 75/635（11.81%）。TRAIN 252 个指令键中有 20 个歧义关系绑定（按关系条目计），Dev 有 8 个；均未用 GT 消歧。首/末重复公开记录的地图元数据差异为 0。
- TRAIN 有效关系类型按状态：near 65、north 10、between 15、along 5、south 5；Dev：near 55、along 5、between 10、north 5。有效关系全部每状态仅一条（between 内两个锚点在描述符中合并成一个参考）。
- 每轮真正监督 **49 行**，4 轮共 196 次状态呈现，非 196 个独立样本；optimizer step 分别 34/31/35/34。fit 日志 loss 从 0.5048 降至 0.2760（按脚本 batch 加权的监控量）。
- reference/value/head 最大梯度范数分别 0.5480 / 1.9456 / 11.8303（裁剪前）；reference 首层权重变化范数 0.1772，输出头权重变化 0.1093。关系分支确实更新。
- **Attention Query/Key 真实梯度始终为 0**：只有一个有效参考时 softmax 恒为 1。其小幅权重变化来自 AdamW weight decay，不能算学到了注意力选择。双参考合成测试梯度非零，说明并非代码断路；真实数据不足以训练多参考选择。
- fit 中排序纠正/破坏 16/1；calibration 为 0/2；Dev 为 2/0。Dev 有 19 个状态改变候选 argmax，1110 个有效 query 分数改变；打乱关系又改变 6 个 argmax 和 1036 个 query 分数。Dev 具名关系状态的平均绝对 delta 为 1.8488，说明不是零头未训练或关系特征完全无作用。

## 为什么排序有改善，最终定位没有改善

Dev 具名关系子集：A0 已正确 70/75（93.33%），原 Native 67/75，Relation 66/75。两次候选排序纠正并未转化为额外 Belief 纠正；Relation 还比 Native 多破坏 1 个原 A0 正确状态。非具名子集 560 状态的 Native 与 Relation 输出相同。结论是**排序略有改善，但监督目标与最终 Belief 定位收益尚不一致**；目前没有实例标注或反事实控制器结果，不能进一步断言具体实例错配/导航原因。

Gate fit 正标签 127/1027（12.37%），calibration 有 30/248 个可纠错机会。3 轮 BCE loss 从 1.4930 降至 0.8750，接受概率仍偏低：fit 最大 0.4804，calibration 最大 **0.46945**，Dev 最大 0.46921；既定阈值候选最小为 0.5，所有候选都接受 0 个，平局规则选 threshold=1.0。不是校准观察到大量误改才拒绝，而是整个分数范围未进入候选阈值区间。Dev 虽有 50 个此固定视觉混合动作可纠错状态，Gate 实际纠正/破坏均 0。

## 主要瓶颈与下一步

1. **具名关系监督稀少，且只有单参考**。仅 17 个 fit 组、49 个有效状态，near 占大头；fit 排序提高而 calibration 退化，泛化证据弱。下一步在 TRAIN 中增加具名方向/多参考组，使用公开别名、类型前缀做保守解析与人工抽检；重复/歧义名称继续无 GT 消歧。按关系组均衡采样；加入方向反事实监督，并把候选 loss 与 TRAIN 的最终 Belief 定位/保真约束联合。先用 TRAIN 独立组验证，确认收益再看内部 Dev。
2. **Gate 训练后的分数与校准网格不匹配，融合没有激活；直接残差也未把排序收益转成定位收益**。下一轮在 TRAIN 内用标签先验初始化 Gate bias，按 TRAIN 校准概率分位数产生阈值（含 0.5 以下），沿用 harm budget，检查正负样本分离而非只看 BCE。另比较“拒绝时保留 Native Residual”的结构，避免将现有 +7 净收益全部退回 A0；是否安全仍需 TRAIN 独立组验证。本轮不降低阈值、不额外加轮数、不按 Dev 调参。

单 seed、状态同组相关，+2 个候选排序正确不足以证明统计稳定提升。当前证据支持“关系分支获得弱排序信号”，不支持“完整方案性能提升”。

## 复现与产物

- `reproduce.sh` 固定 seed=0、顺序执行 smoke/full，使用独立输出目录；训练命令为 `python -m scripts.train_native_relational_grounding`。
- `smoke/`、`full/` 保存 protocol、训练 history、TRAIN 校准、Dev 指标、gradient audit、complete、日志；`full/diagnostic.json` 保存只读诊断。`public_relation_audit.json` 保存公开关系统计。
- 派生权重仅保存在本工作树 `runs/relation_gate_smoke_s0/pilot.pt` 与 `runs/relation_gate_full_s0/pilot.pt`，未替换任何原权重，且 `admitted_for_runtime=false`。full SHA256 `1300268a23b9827d822290dc70c0aabfef10e41f9b088496bae846c628d6489d`；smoke SHA256 `45b1a18a65defd19e5730fa1ed04c70b84c0161d7ca269cdd32aaa5c742868bf`。
- `protocol.json` 的 implementation_commit 是训练开始时 HEAD 基点，具体带修复的源文件由 implementation_sha 标识，最终代码随本诊断分支提交；没有把基点误当成已包含修复的提交。
