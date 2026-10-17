# PolyBelief 论文实验补充 TODO

更新：2026-10-17。结果来自实际文件和逐 Episode 验证；Done 只表示本行明确范围完成，不代表全部严格baseline复现。
正文方法统一称PolyBelief；旧源路径和文件内容保持原样以便溯源。正式冻结权重与三split结果未改。
2027-CVPR 是多仓库容器，新增代码/报告位于当前 research/visual-residual-20261013 工作分支；不修改其他研究工作树。
本轮无新训练或推理，完成已有资产重算与审计。没有活跃实验可据此标为Running，旧队列状态不作完成依据。

字段：数据集的“全部”指Val-Seen / Val-Unseen / Test-Unseen；数值比例在CSV均为百分比，NE为米。


## P0

|experiment|status|dataset|source|result_path|missing|acceptance|estimated_cost|
|---|---|---|---|---|---|---|---|
|冻结PolyBelief完整模型三split复核|Done|Val-Seen / Val-Unseen / Test-Unseen|冻结checkpoint；现有完整轨迹|reports/paper_experiments/verification.json|无；本轮逐条重算与正式CSV一致|2470/2697/5281完整ID、NE/SR/OSR/SPL吻合|CPU离线数分钟；已完成|
|Table1原论文Reference及全方法缺口|Done|全部|SBFNav Table1；官方资源|reports/paper_experiments/table1_paper_reported.csv|严格本地复现仍有缺口，见各方法条目|11个方法三split有值或明确缺失；Human只作上限|离线；已完成|
|Table1本地共同指标与协议分表|Done|全部|已有metrics及逐Episode轨迹|reports/paper_experiments/table1_ours_unified.csv|未完成方法留空；不表示全部baseline完成|本地值与论文值分离，所有协议偏差保留|离线；已完成|
|官方Easy/Medium/Hard及Test离线统计|Done|Test-Unseen|CityNav发布子集ID；既有预测|reports/paper_experiments/difficulty_analysis.md|论文未发布阈值生成程序，准确使用官方清单|互斥且覆盖5281；全部ID、源SHA及计数核验|CPU数分钟；已完成|
|Polygon vs Centroid vs Bounding Box闭环|Done|Val-Seen / Val-Unseen|新实验：固定Train-Dev、3seed、独立初始化|reports/paper_experiments/polygon_ablation/POLYGON_ABLATION_REPORT.md|无；诊断与重训两口径分别报告，结果正负均保留|三组同清单、同控制器和选模规则；静态结果另表|仅推理约3×15–40分钟；若需独立训练另估，不在本轮启动|
|Geometry-only vs Geometry + Visual Belief|Done|Val-Seen子集|正式冻结模型Seen128|reports/paper_experiments/ablation_results.csv|只有Seen128正式权重干预；全量双validation尚未补齐|当前完成范围明确为128条；后续扩展须匹配权重与样本|已完成；扩展约2×15–40分钟|
|Calibrated Visual Residual四组完整泛化|Done|Val-Seen / Val-Unseen|相同已训练calibrated_residual/best.pt|VISUAL_RESIDUAL_UNSEEN_VALIDATION.md|Test未运行且无需用Test选候选|四组2470/2697；逐条配对、Zero一致|已有全部结果，禁止重跑|
|视觉残差地图级Bootstrap|Done|Val-Seen / Val-Unseen|既有20,000次paired bootstrap|visual_residual_unseen_validation/bootstrap_results.json|仅4张Unseen地图；主比较SR CI下界=0，不能称明确跨地图正收益|固定seed；episode加权及地图等权；CI如实保留|已完成|

## P1

|experiment|status|dataset|source|result_path|missing|acceptance|estimated_cost|
|---|---|---|---|---|---|---|---|
|Direct-to-Goal / Learned Trajectory / Guarded-Blend正式权重对照|Pending|Val-Seen / Val-Unseen|历史controller Seen128结果可复用|reports/paper_experiments/ablation_results.csv|历史头不同；尚无正式冻结模型三组全量受控比较|同正式权重和ID、只改变执行方式；先核查已有配置接口|仅推理约3×15–40分钟；不训练|
|Stop Head / Arrival Guard受控对照|Done|Val-Seen|冻结完整模型；六种guard全Seen|reports/paper_experiments/ablation_results.csv|Unseen扩展未做；不据Test选guard|全Seen六组2470，stop0.3固定，保留终止原因|已有；不重跑|
|Oracle Goal闭环瓶颈诊断|Pending|Val-Seen / Val-Unseen|已有候选覆盖oracle不是目标控制oracle|reports/paper_experiments/asset_inventory.csv|未核验真正GT目标替换的闭环；须明确仅诊断、不能作为方法输入|单独标注oracle权限，固定预算；严禁正式网络输入GT|推理约2×15–40分钟；先协议审查|
|Current RGB vs Current + History|Done|Val-Seen子集|正式冻结模型Seen128|reports/paper_experiments/ablation_results.csv|全量验证缺失，不能将早期不同stage权重混入|同权重128条成对；明确完成范围|已有；全量扩展约2×15–40分钟|
|28x28 vs384x384目标网格|Done|全部|原冻结权重、完整三split|full_resolution_384/COMPARISON_28_VS_384.csv|尚无384与独立校准残差联合评测，严禁收益相加|三split完整；只改变目标网格；固定stop0.3|已有全部结果；不重跑|
|官方目标距离难度分层|Done|Test-Unseen|官方难度ID|reports/paper_experiments/table2_difficulty_comparison.csv|自定义距离分箱若需使用必须另表，不能冒充官方难度|使用发布难度清单和完整ID|已完成|
|语言关系类别细分|Pending|Val-Seen / Val-Unseen|现有语言解析与原指令|reports/paper_experiments/asset_inventory.csv|没有完整闭环类别统计与经核验标签；不可把静态关系诊断当闭环|先固定解析规则/人工抽查，多标签策略及分母明确|CPU分析约0.5–2小时，无训练|
|参数量/每步推理/峰值显存|Done|Val-Unseen|2697条完整HETT-style profiling|reports/paper_experiments/efficiency_existing_evidence.json|只适用于原28模型；baseline同硬件profiling尚缺|完整样本、warmup/计时边界、参数组件及峰值定义保留|已有；不重跑|
|GeoNav严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Original-data DeepSeek Test completed; GPT-4o access/configuration not verified; Seen/Unseen absent. Paid service cannot be assumed.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|约5–50k模型调用；费用/时长依API，当前不可可靠估计|
|HTNav严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Two local repaired configurations have validation results; official weights not found in released README; local adaptation differs from official method.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|已有validation不重跑；严格复现缺作者配置/权重；若重训参考旧10epoch约15小时以上|
|HETT严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Official checkpoint archive exists and SHA verified; HETT/main is a modified trajectory-belief research checkout, not a verified checkpoint-compatible upstream policy. Isolated --help passed after dependency fix; hardcoded paths/output and missing episode export require an upstream-compatible isolated evaluator. Historical document only.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|上游兼容源码恢复/导出约1–4小时；全量推理约1–6 GPU小时，需smoke测定；不能直接使用当前研究策略|
|FlightGPT严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Official public weight identified (~16.6GB); local four shards absent; native evaluator adaptation and model-load smoke not completed.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|下载16.6GB，1–4小时适配/加载检查；全量VLM推理预计数小时至数天，须实测|

## P2

|experiment|status|dataset|source|result_path|missing|acceptance|estimated_cost|
|---|---|---|---|---|---|---|---|
|Seq2Seq+GSM严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Historical aggregates exist; local checkpoint references broken; exact source/log/trajectory chain incomplete.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|精确GSM训练权重缺失；恢复优先，不重新训练|
|CMA+GSM严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Historical aggregates exist; local checkpoint references broken; exact source/log/trajectory chain incomplete.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|精确GSM训练权重缺失；恢复优先，不重新训练|
|MGP严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Official checkpoint recovered in isolated paper_baseline_assets directory (SHA in mgp_recovered_asset.json); not proven identical to missing historical weight. Saved per-episode historical traces/source chain unavailable for common-evaluator replay; native aggregates remain separate.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|228MB官方权重已恢复；源码/输入/历史身份链路核查约0.5–2小时；若需新共同协议推理数小时，需smoke|
|AerialVLN+GSM严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Local original-data adaptation completed; different data/action/imagery protocol; aligned full training incomplete.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|对齐版全量5epoch预计数十GPU小时，当前未完成，不本轮重训|
|Random严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Historical native policies exist; no saved full-split trajectories with complete provenance for this frozen observed-only protocol.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|本地固定策略CPU数分钟；论文随机策略不明确，禁止fit|
|SBFNav严格本地复现缺口|Blocked|Val-Seen / Val-Unseen / Test-Unseen|官方资源；具体见baseline_protocol_audit.md|reports/paper_experiments/table1_local_other_protocols.csv|Independent reimplementation has trajectories; validation epoch14 vs Test epoch12; old selection used Val-Unseen, incompatible with new selection constraint.|代码/权重/数据SHA/逐Episode/日志/命令/冻结选模完整，保持方法原身份|严格Seen-only重新选模需新训练/作者权重，历史Unseen选模不能洗白|
|统一baseline多种子与同硬件效率|Pending|Val-Seen / Val-Unseen|先完成单种子严格复现|reports/paper_experiments/missing_results.md|部分历史Random为5seed均值，不能与单seed不加标识混排|按预先预算多seed；同硬件测量；不据Unseen/Test选择seed|约单seed预算×种子数；本轮不启动|

<!-- POLYGON_ABLATION_UPDATE -->

## Polygon独立贡献实验进度

状态：Done。新静态结果62行，导航结果62行。独立报告位于reports/paper_experiments/polygon_ablation/POLYGON_ABLATION_REPORT.md；冻结干预与3seed公平重训、Descriptor Only与End-to-End严格分开。

|task|status|evidence|
|---|---|---|
|几何通路审计、三种表示、无旁路/容量/缓存测试|Done|POLYGON_FEATURE_AUDIT.md; tests.log|
|冻结权重完整静态干预|Done|24次实际完整评估+2个Full输入相同的scope别名|
|两口径3Seed公平重训|Done|15/15独立训练；training_status.csv及training/*/complete.json|
|所有重训权重Seen/Unseen完整静态评估|Done|36/36逻辑结果行|
|原始/优化执行路径Seen128精确一致性|Done|rendering_parity.json（完成前不允许声称通过）|
|同一冻结Trajectory/Stop全量闭环|Done|62/62逻辑结果行；54次实际全split评估，Full别名另计|
|三Seed配对、地图簇CI及LOMO和分组分析|Done|paired_statistics.json; seed_summary.csv; geometry_subgroup_results.csv|
|原始结果和权重远程备份|Done|frozen_remote_sync.json为冻结阶段已验证归档；最终remote_sync.json另记|

<!-- GEOMBELIEF_REMAINING -->

## GeomBelief independent contribution experiments

Archived Polygon baseline 34d2974 retained. New complete metric rows: 13/84 (includes reused controls). See [map geometry report](../reports/paper_experiments/geombelief_remaining/MAP_GEOMETRY_ABLATION_REPORT.md) and reports/paper_experiments/geombelief_remaining/experiment_manifest.json for exact paths, SHA, missing experiments and paired statistics. P1 fair visual-input retraining remains Pending until explicitly completed; existing Real/Zero/Shuffle/No-belief results are reused.
