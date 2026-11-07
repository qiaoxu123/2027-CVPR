# 候选确认监督审计

2026-11-07。读取历史G5核验原始文件，未修改标签、权重或评测协议。

两个版本合计只有1条人工确认的可见修正框、1条不确定记录；草稿不作为已确认监督。该数量不足以训练并独立评价实例排序器。200条Val候选审核页仍没有回传人工标签，不能将历史TRAIN标签与其混为一谈。

唯一可见修正框记录的2048px图像中高度约5.75px，且存在384源图与2048审核图元数据。复用前需要重新核验坐标兼容与实例边界，不能直接当作可靠像素金标准。文件来源与SHA见audit.json。

代码核查：graphnav/trajectory_policy.py的trajectory_losses使用当前位置距真实目标≤20m且候选距目标≤20m作为Stop正标签，错误候选为负标签。因此上一轮128条均在首候选停止，不能解释为训练目标本来就要求到任意候选停止。下一项诊断应比较Stage1训练候选上下文、Geometry Checkpoint与当前V2部署候选的分布及绑定一致性；优先复用TRAIN缓存，不调Stop阈值，不在Val重复选参。

本轮没有新增导航成绩。完整Val-Seen基线仍为48.54%，55%以上目标未达成。

## Controller与Geometry权重来源核对

实际加载Stage1 best.pt与当前部署V2 Checkpoint：Stage1记录为Stage C，代码configure_stage在该阶段同时解冻a3与trajectory。当前运行器只提取该Checkpoint的trajectory权重，Geometry则来自另一个V2 Checkpoint。

70个共同Geometry张量中，50个浮点张量不相同；语言投影、token attention、pair geometry相关层均有变化，具体相对L2差见controller_geometry_comparison.json。这是候选上下文分布失配的具体来源，不能仅据权重差认定因果。模型维度仍匹配（hidden=128、grid=28、Top-K=20）。

建议下一步在TRAIN固定物理目标留出中，保持候选世界位置相同，分别送入Stage1原始上下文与当前V2上下文，比较错误候选到达时Stop输出及正确候选接受率；先检验跨Checkpoint接口假设，再决定是否适配控制头。原Stage1曾按Seen128选择，需明确其历史验证暴露，不能把同一Pilot作为独立泛化证明。

## 真实GPU上下文反事实（128条TRAIN development）

冻结Stage1全部权重，固定当前V2的20个候选位置、分数、熵；仅替换相同网格位置的dense context。每个候选构造到达位置（固定yaw=0、相对地面30m），共2560个反事实状态，非真实闭环或观测序列。GT仅在公开前向完成后确定候选是否距目标≤20m。

|上下文|正确候选接受率|错误候选接受率|
|---|---:|---:|
|当前V2|96.21% (897正例)|92.48% (1663负例)|
|Stage1原Geometry|95.88%|91.28%|

原上下文仅降低错误接受1.20pp，仍大量接受错误候选。因此不支持直接通过恢复原上下文解决候选确认。当前Stop输出不能作为可靠实例确认信号。此结果限于固定高度、yaw及候选到达反事实，不代表轨迹全状态或SR；128条同一任务的20候选不能作为2560个独立任务。

复现脚本：scripts/probe_v2_controller_context.py。公开前向结果保存在服务器context_probe128/public_probe.json；指标见context_probe128_metrics.json。未训练、未运行Val/Test；应继续寻找具有目标身份信息的候选验证监督，而非调整Stop阈值。

## 完整Cambridge development确认

使用同一冻结协议扩展到全部3126条任务、62520个候选到达反事实状态：19602正位置代理、42918负位置代理。当前V2上下文的正确接受率99.37%、错误接受率98.36%、AUROC 0.67755；Stage1原上下文分别为99.29%、98.19%、0.67554。恢复原上下文的错误接受率仅下降0.172pp，AUROC未改善。

该结果不支持“恢复Stage1原上下文即可解决候选确认”的假设。Stop分数存在有限位置相关区分信号，但原0.3阈值下几乎接受全部候选到达状态，不能用于当前顺序候选验证。此处不选择新阈值，不运行新的Val Pilot。下一项模型工作应针对真实目标身份/语言匹配构建候选确认监督，不能继续把到达候选当成确认目标。

完整前向已真实GPU执行；未训练。逐候选公开输出保存在服务器context_probe_full_dev/public_probe.json，复现使用probe_v2_controller_context.py --limit 0。统计为配对TRAIN机制诊断，不是独立场景泛化或真实闭环成绩，未按候选数夸大独立任务数量。

## 冻结Stage1图文Adapter补充探针

复用512条TRAIN当前起点ROI缓存，保持相同图像、文本及候选。仅比较训练过的image/text residual Adapter，尺度均匀平均，不启用quality gate；不宣称复现完整Stage1匹配器。模型完成打分后才用GT评分。

26条同时含支持正确位置及至少2个支持候选的任务中，Geometry首个可观测候选命中20/26；Adapter在384 ROI上完整指令/外观描述均15/26，与此前原始余弦15/26相同；native-GSD ROI均13/26，未超过原始余弦15/26、16/26。结果为20m位置代理，非实例识别准确率。

该冻结Adapter不能补足当前视觉排序能力，不进入Val Pilot。无需重复训练旧Adapter并把它当作新方向。脚本probe_cached_roi_adapter.py与逐任务输出frozen_adapter_probe.json可复现；未训练或更改权重。
