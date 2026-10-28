# A2修订版：实现与训练准备

目标：在保持原A0、冻结骨干、控制器及28×28网格的前提下，迭代内部验证并最终完成A0/A2全Seen2470与Unseen2697确认。尚未实现该目标。

旧A2完整20+5训练和校准结果保留；旧门控不启用。新版本代码为revision_*，与旧model/train/fusion/labels分开。

新版本：可靠跨类别负例＋全候选空间定位监督；输入当前A0 Belief；分别学习候选可信度与更新收益；采用最大25%全局概率混合并保留精确OFF回退；偏移为0。已预留256个原dev/calibration中从未采样的目标，未采新观测，不参与任何新训练负例。原数据采集继续停止。

原TRAIN31415/dev4618快照重用。数据审计记录32,408个新增可用跨类别负例。TRAIN-only正确候选oracle，在同一.25上限下新融合修正1713、破坏0，旧局部域融合只能修正30；这仅证明结构容量，不是训练后或闭环收益。

完整CPU测试340项通过；独立审查复核18项revision测试，以及50种随机batch的旧逐项/新按批loss和梯度一致性（loss差≤2.384e-7，梯度差0）。真实TRAIN CUDA已通过，rank/language/trust/category/benefit均有非零有限梯度；按批改写后稳态batch8约0.020秒，原逐观测检查约0.25秒。正式训练尚待最终全行数据审计和启动状态确认。

设计：docs/superpowers/specs/2026-10-28-a2-revision-design.md；计划：docs/superpowers/plans/2026-10-28-a2-revision-implementation.md。
