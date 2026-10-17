# PolyBelief 论文实验数据与补充总结

本轮完成已有资产Audit、离线Reuse与可立即得到的Complete数据整理；外部严格Reproduce仍按真实资源阻塞，不重训全部baseline。
所有结果均区分原论文报告、本地共同XY指标重算、本地其他协议、历史仅聚合/文档。
本轮额外恢复MGP官方228MB权重并保存SHA，未覆盖旧权重；HETT隔离补依赖后help可运行，但当前代码含研究结构改动，不能冒充官方HETT执行评测。

## 正式冻结完整模型

|method|split|episodes|NE|SR|OSR|SPL|checkpoint_sha256|
|---|---|---|---|---|---|---|---|
|PolyBelief|val_seen|2470|29.1192|47.6113|55.6680|46.0028|00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a|
|PolyBelief|val_unseen|2697|51.5327|22.5436|31.7019|21.8205|00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a|
|PolyBelief|test_unseen|5281|35.0079|37.1521|46.5632|35.8061|00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a|

本轮逐Episode坐标重新计算四指标，与正式三split CSV及原metrics吻合，未覆盖正式结果或权重。
方法名称暂统一PolyBelief，数值身份仍为原冻结完整模型；独立训练校准残差和384评估均是消融，不能替换其中任一split凑主表。

## Table 1 与数据审计

table1_paper_reported.csv保留所有目标baseline与Human参考，并额外保留HETT*/SBFNav*两修订数据行。PolyBelief原论文不存在，留空注明原因。
table1_ours_unified.csv保留共同revised数据+XY evaluator的本地值；原生控制器/历史选模与本地适配身份保留，不宣称同输入或严格复现。
SBFNav的epoch12有三split完整结果，epoch14仅验证另列，不择优拼表。HTNav原生SPL与欧氏最优距离定义略有差异，两个版本分别保存，不覆盖原值。
table1_local_other_protocols.csv收录原始数据、模型替换和历史记录。不存在的实验不填数值。
特别发现：SBFNav称删除331条，本机发布数据实际删311条；没有作者文件SHA，不能断言修订文件完全相同。

## Table 2

|method|difficulty|episodes|NE|SR|OSR|SPL|
|---|---|---|---|---|---|---|
|PolyBelief|easy|2081|33.8038|37.7222|51.0812|35.2982|
|PolyBelief|medium|1678|36.2374|34.6246|42.1335|33.8712|
|PolyBelief|hard|1522|35.2988|39.1590|45.2694|38.6337|

|difficulty|episodes|delta_SR_pp|delta_SPL_pp|
|---|---|---|---|
|easy|2081|6.8717|15.2601|
|medium|1678|6.0787|10.6803|
|hard|1522|6.8988|9.8092|

上述差值相对本地SBFNav独立重实现epoch12；受其历史Unseen选模与控制器差异限制，不等同严格方法归因。完整难度统计和original论文表独立保存。

## 核心机制结论

Calibrated Visual Residual 已有四组完整Seen2470/Unseen2697，不重复运行。
Unseen Real−No-belief SR +1.8168pp，整地图cluster95%CI [0,3.4410]pp；SPL +1.2329pp，CI [-0.6561,2.7960]pp。
Zero与No-belief逐条行为一致。相对Shuffle的地图级SR/SPL区间为正，支持利用RGB对应信息；但仅4张独立Unseen地图，主比较CI下界=0，不能报告跨地图明确正收益。
Seen的地图级主比较同样跨零，不能沿用仅episode bootstrap的强结论。

Stop/Arrival Guard六组完整Seen、Current RGB/History的Seen128、全三split28/384及2697条Unseen profiling均复用。
缺少同正式权重的Polygon/Centroid/Bounding Box、三控制器、Oracle Goal和完整关系类别统计，按TODO保留Pending，不把早期静态/其他权重结果写成完成。

## 复现和资产

离线命令：在本工作树用Baselines/.reproduction-env/bin/python运行 `-m scripts.build_paper_experiments`。
experiment_registry.csv逐行列出结果路径、数据SHA、Checkpoint SHA、源commit、原推理重建命令、预测和日志存在性。
build.log/build_provenance.json记录本次统计；sources/manifest.json记录外部原始网页及SHA；repository_inventory.json记录分支/工作树/dirty状态。
asset_inventory.csv/checkpoint_inventory.csv为扫描清单，selected_checkpoint_inventory.csv对使用到的权重验SHA。未恢复的旧权重明确missing，不凭文件名替代。
verification.json记录逐Episode、分母和聚合校验。difficulty_episode_manifest.json保留官方清单到本地ID映射。

common_episode_metrics.csv保存本次可重算来源的逐Episode共同指标，通过experiment_id连接registry；原预测轨迹位置保持原地。
新增结果是可重建的离线产物，原始大规模轨迹与权重保持原地，Git提交源代码、报告、数据索引和SHA，避免复制共享DATA或checkpoint。
完整待办见 [PAPER_EXPERIMENT_TODO.md](../../../docs/PAPER_EXPERIMENT_TODO.md)，机器可读todo_status.csv同时保存。

## 已有完整Val-Unseen效率数据

2697个Episode；每决策计时样本22144个。策略参数1,053,290，含冻结BERT和SigLIP双塔的驻留参数992,389,996；包含未激活的旧分支。峰值allocated显存3.7825GiB。

|component|min|mean|population_variance|
|---|---|---|---|
|rollout_ms|31.1567|48.1659|595.4020|
|language_gpu_ms|0.0000|3.0656|72.9394|
|vision_gpu_ms|11.2031|13.1334|1.4599|
|static_graph_gpu_ms|0.0000|0.3459|4.0593|
|visual_residual_history_gpu_ms|2.1565|3.4449|0.2363|
|trajectory_stop_gpu_ms|3.9414|4.3426|0.1939|
|model_inference_gpu_ms|17.4673|24.3324|110.0368|
|environment_ms|3.9183|6.7220|110.2976|
|other_rollout_ms|9.1427|17.1114|140.5101|

上述均值单位ms，方差ms²；GPU事件跨度和端到端wall time分别保留。完整计时边界、并发进程及首次冷启动规则见efficiency_existing_evidence.json和原launch记录；不据此冒充不同baseline的同机速度对比。

## 远程备份

代码和报告已同步到research/visual-residual-20261013；新增四组完整Unseen逐Episode轨迹及恢复的MGP权重另存[远程资产](https://github.com/qiaoxu123/2027-CVPR/releases/tag/polybelief-paper-experiments-20261017)。17个文件，压缩包205203635字节；GitHub服务端SHA与本地一致，下载manifest复核通过。校验与恢复清单见remote_sync.json、POLYBELIEF_ASSET_MANIFEST.json及POLYBELIEF_SHA256SUMS。原冻结权重继续使用旧release，不重复上传。

<!-- POLYGON_ABLATION_UPDATE -->

## Polygon独立贡献补充

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
