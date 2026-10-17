# PolyBelief Test-Unseen 官方难度分析

采用 CityNav 发布的 `citynav_test_unseen_easy/medium/hard.json` 的精确 Episode ID 成员关系。
修订版为官方原始成员集合与 HETT refined 全集的交集。原始 5311 条，修订版 5281 条。
ID = (map_name, object_id, ann_id)，不按行号或描述文本连接。三个集合已验证无重复、互斥、完整覆盖；源文件 SHA 和全部 ID 在 difficulty_episode_manifest.json。

SBFNav 正文只说明沿用难度分组，并引用 OpenFly；HTNav §4.1 明确说基于起点到目标直线距离。
已发布 CityNav loader 直接按文件后缀加载集合，没有公布生成阈值程序。因而本次准确复用发布的成员清单，不自行发明距离阈值。
源清单中记录的 dist_start_to_target 的实际范围仅是核验性描述，不能解释为新分组规则。

参考：[SBFNav Table 2及数据说明](https://arxiv.org/html/2609.05841v1)，[HTNav §4.1](https://arxiv.org/html/2604.08883)，[CityNav官方loader](https://github.com/water-cookie/citynav/blob/main/gsamllavanav/dataset/mturk_trajectory.py)。

## 同一修订版集合、共同 XY 指标的本地结果

|method|variant|difficulty|episodes|NE|SR|OSR|SPL|
|---|---|---|---|---|---|---|---|
|PolyBelief|frozen full model; C epoch1; 28x28|easy|2081|33.8038|37.7222|51.0812|35.2982|
|PolyBelief|frozen full model; C epoch1; 28x28|medium|1678|36.2374|34.6246|42.1335|33.8712|
|PolyBelief|frozen full model; C epoch1; 28x28|hard|1522|35.2988|39.1590|45.2694|38.6337|
|SBFNav|paper-based reimplementation epoch12|easy|2081|37.0201|30.8506|53.8203|20.0381|
|SBFNav|paper-based reimplementation epoch12|medium|1678|38.7105|28.5459|47.5566|23.1908|
|SBFNav|paper-based reimplementation epoch12|hard|1522|38.0063|32.2602|47.1091|28.8245|
|HTNav|repaired released code;reproduction_20261012|easy|2081|154.2419|7.5444|14.8486|7.1616|
|HTNav|repaired released code;reproduction_20261012|medium|1678|170.6269|5.0060|9.2968|4.6560|
|HTNav|repaired released code;reproduction_20261012|hard|1522|138.6242|5.0591|7.8187|4.7384|

## PolyBelief 与本地 SBFNav 的差距

|difficulty|episodes|PolyBelief_SR|SBFNav_SR|delta_SR_pp|PolyBelief_SPL|SBFNav_SPL|delta_SPL_pp|
|---|---|---|---|---|---|---|---|
|easy|2081|37.7222|30.8506|6.8717|35.2982|20.0381|15.2601|
|medium|1678|34.6246|28.5459|6.0787|33.8712|23.1908|10.6803|
|hard|1522|39.1590|32.2602|6.8988|38.6337|28.8245|9.8092|

这里是已完成的 SBFNav 独立论文重实现 epoch12 对照，不是作者发布权重。
epoch12 的三个 split 均有完整结果，并与 epoch14 的验证结果分开列出；不能跨权重拼表。
历史 checkpoint 选择使用 Val-Unseen，不能视作满足 Seen-only 选模的严格基线。
本轮只离线重算，未依赖 Test 指标调整模型。PolyBelief 在不同难度的变化请以表中实值为准；难度不是语义复杂性或指令长度标签。
PolyBelief的SR从Easy的37.72%降至Medium的34.62%，Hard反而为39.16%；SPL分别35.30%、33.87%、38.63%。
因此性能没有随官方难度单调下降。Hard的SPL更接近SR也受到较长最优路径在SPL归一化中的影响，不能单独解释为语义推理增强。
相对本地SBFNav，三个难度SR均有约6.08–6.90pp的点估计优势；未经配对地图级检验且控制器不同，不宣称这些差距是统计显著的模块收益。

## 原论文结果与其他本地协议

table2_paper_reported.csv 独立保存 SBFNav Table 2 引用值：论文使用 original，不能与上述 revised 直接作受控收益归因。
其中 Seq2Seq/CMA 没有 +GSM，不得把它们替换成 Table 1 的对应方法。
table2_difficulty_comparison.csv 另包含原始数据 AerialVLN+GSM、GeoNav 的已有本地结果，以及历史 GSM 难度汇总，均有 result_type 标识。
没有逐 Episode 证据的历史汇总只保留其原值，不能宣称重新验证了每个 ID。
