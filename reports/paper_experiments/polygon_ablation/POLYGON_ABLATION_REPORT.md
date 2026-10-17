# PolyBelief Polygon Geometry 独立贡献验证

状态：部分完成；未完成项目保留Pending。[信息通路审计](POLYGON_FEATURE_AUDIT.md)与frozen_plan.json先于实验结果产生。
独立训练已完成8/15；当前结果行数：静态26、导航0，包含明确标记的Full跨口径相同输入别名。最新运行阶段另见progress.json及workflow.log。
主对照是End-to-End独立重训；Descriptor Only仅解释描述符增量，冻结干预仅解释已训练权重的依赖。
相同10-D/模型容量、3个种子、15epoch、batch16、AdamW5e-5。训练样本按Train对象分组保留固定Dev，选择只用Dev R@1及NE；未用Unseen/Test选模或修改配置。
全部模型参数727696，其中优化参数453517；保留相同未激活GAT/视觉分支的结构，但不执行GAT消息传递或RGB推理。matched_training_checks.json逐Seed核验初始化、容量及每Epoch数据顺序。
Full Polygon在两个口径的输入完全相同，每seed按同一SHA复用；共18个逻辑条件、15次独立训练。seed0预先指定优先闭环，并保留seed1/2所有结果。
所有静态评估完成标志all_static_complete.json先于任何闭环。静态Seen2426/Unseen2679，导航Seen2470/Unseen2697；原始定位/导航结果未冒充本实验结果。

指标沿用现有evaluator：Recall命中半径20m；SR为最后位置到GT的XY距离≤20m，不额外要求policy_stop；OSR为路径任一点命中；SPL为SR乘起点–目标直线距离/max(实际XY路径长度,起点–目标距离)。NE为最终XY距离（米）。False_Stop_Rate为失败policy_stop数/policy_stop总数，无policy_stop时为null；另保留失败stop数/全部Episode的false_stop_episode_fraction，不能把无主动stop当作0%错误停止。以上GT均只用于loss或推理完成后的评分。

## 三个最终问题

1. Full Polygon是否比Centroid/Bounding Box更好：以下End-to-End重训配对效应给出实际方向，不预设正收益；统计区间跨零或点估计为负时，不能写成普遍优势。
2. Seen/Unseen是否一致及统计支持：分别报告；4张Unseen地图仅为4个独立地图簇，主要依据整地图cluster CI和Leave-One-Map-Out，不能以Episode CI替代。
   簇按官方map_name地图块定义，并非4座独立城市；不能把该结果扩写成跨城市泛化结论。
3. 来源是边界/Inside/Extent还是旁路：Descriptor Only的Full与BBox只改变边界/Inside，Extent相同；End-to-End还改变prior/rules/relation_fields及节点统计。冻结No Inside仍保留signed boundary符号；No Inside Information另清除符号。仅冻结敏感性不能证明某模块公平重训后的独立因果贡献。

- 公平重训及完整闭环尚未全部完成，不能从冻结特征敏感性直接判断表示优劣。

冻结权重通路敏感性（Full−干预，负值表示干预后反而更好，不隐藏）：

|split|intervention|delta_R1_pp|map_ci95_pp|
|---|---|---|---|
|val_seen|no_boundary|5.688375927452597|[2.788491308054306, 9.018595873395943]|
|val_seen|no_inside|-0.4122011541632316|[-1.8454251989209483, 1.1294549499307376]|
|val_seen|no_inside_information|-0.6183017312448474|[-1.3291743516819856, 0.0]|
|val_seen|no_extent|-0.24732069249793898|[-0.9383593367957496, 0.5140155309221562]|
|val_seen|prior_to_centroid|1.9785655399835118|[0.38843576043844485, 3.571428571428571]|
|val_seen|rules_to_centroid|-0.08244023083264633|[-0.3927789802996052, 0.21962334883271922]|
|val_seen|nodes_to_centroid|1.030502885408079|[0.17248628273185532, 1.8828527645165654]|
|val_unseen|no_boundary|5.375139977603583|[1.8318068276436303, 8.254397834912043]|
|val_unseen|no_inside|-0.41060097051138483|[-0.832639467110741, 0.34965034965034963]|
|val_unseen|no_inside_information|-1.0451661067562523|[-2.426448286320898, 0.030303030303030304]|
|val_unseen|no_extent|-0.6345651362448675|[-1.0923535253227408, 0.11655011655011654]|
|val_unseen|prior_to_centroid|-0.2986188876446435|[-1.7526455026455026, 2.097902097902098]|
|val_unseen|rules_to_centroid|-0.2612915266890631|[-0.3942978465271459, -0.11210762331838565]|
|val_unseen|nodes_to_centroid|0.0|[-0.9099181073703366, 0.6655574043261231]|


## 冻结权重完整静态结果

这些是当前权重的输入干预，不能代替公平重训结论。完整逐样本预测与配对变化保留。

|scope|representation|split|samples|R@1|R@4|R@8|NE|
|---|---|---|---|---|---|---|---|
|descriptor_only|bounding_box|val_seen|2426|0.46949711459192084|0.6723000824402309|0.7765869744435284|29.488187087314344|
|descriptor_only|bounding_box|val_unseen|2679|0.22172452407614782|0.3803658081373647|0.4916013437849944|51.38930784120805|
|descriptor_only|centroid|val_seen|2426|0.4167353668590272|0.6112943116240726|0.7333058532563891|33.552677345909764|
|descriptor_only|centroid|val_unseen|2679|0.1709593131765584|0.3176558417319896|0.4135871593878313|58.257374373132855|
|descriptor_only|full_polygon|val_seen|2426|0.4703215169002473|0.6859027205276175|0.7802967848309975|29.35631595771861|
|descriptor_only|full_polygon|val_unseen|2679|0.22209779768570362|0.389324374766704|0.49981336319522207|51.9385775548391|
|descriptor_only|no_boundary|val_seen|2426|0.4134377576257213|0.6170651277823578|0.7308326463314098|33.34613400447103|
|descriptor_only|no_boundary|val_unseen|2679|0.1683463979096678|0.3124300111982083|0.4158268010451661|58.55248492880516|
|descriptor_only|no_extent|val_seen|2426|0.4727947238252267|0.6838417147568013|0.7819455894476505|29.396742128234997|
|descriptor_only|no_extent|val_unseen|2679|0.2284434490481523|0.3915640164240388|0.5016797312430011|51.769615369129646|
|descriptor_only|no_inside|val_seen|2426|0.47444352844187965|0.6879637262984336|0.782769991755977|29.060094899120944|
|descriptor_only|no_inside|val_unseen|2679|0.22620380739081747|0.39455020530048523|0.5054124673385592|51.32555580415046|
|descriptor_only|no_inside_information|val_seen|2426|0.4765045342126958|0.685078318219291|0.7786479802143446|29.108529021375354|
|descriptor_only|no_inside_information|val_unseen|2679|0.23254945875326613|0.3975363941769317|0.5162374020156775|51.687409684390175|
|descriptor_only|nodes_to_centroid|val_seen|2426|0.4600164880461665|0.6723000824402309|0.7790601813685079|29.631226209943588|
|descriptor_only|nodes_to_centroid|val_unseen|2679|0.22209779768570362|0.3814856289660321|0.49832026875699886|53.07160165208451|
|descriptor_only|prior_to_centroid|val_seen|2426|0.4505358615004122|0.6817807089859852|0.7802967848309975|29.62495038200781|
|descriptor_only|prior_to_centroid|val_unseen|2679|0.22508398656215006|0.4001493094438223|0.5218365061590146|51.75259106011923|
|descriptor_only|rules_to_centroid|val_seen|2426|0.4711459192085738|0.6892003297609234|0.7823577906018137|29.223253046370495|
|descriptor_only|rules_to_centroid|val_unseen|2679|0.22471071295259426|0.3896976483762598|0.5020530048525569|51.61414480231599|
|end_to_end|bounding_box|val_seen|2426|0.45383347073371805|0.6694146743610883|0.7753503709810388|29.70868491266546|
|end_to_end|bounding_box|val_unseen|2679|0.22284434490481522|0.3926838372527062|0.49570735349010825|50.722553726584664|
|end_to_end|centroid|val_seen|2426|0.42126957955482275|0.6174773289365211|0.72959604286892|32.7037455541038|
|end_to_end|centroid|val_unseen|2679|0.18066442702500934|0.3340798805524449|0.4419559537140724|57.664070723371836|
|end_to_end|full_polygon|val_seen|2426|0.4703215169002473|0.6859027205276175|0.7802967848309975|29.35631595771861|
|end_to_end|full_polygon|val_unseen|2679|0.22209779768570362|0.389324374766704|0.49981336319522207|51.9385775548391|


## 三Seed均值及样本标准差

比例列均为0–1，NE为米；delta_pp及map_ci95_pp明确使用百分点。每seed原始结果分别保留在两个results.csv中，不选择最佳seed。

|endpoint|scope|representation|split|seeds|samples_per_seed|R@1_mean|R@1_std|R@4_mean|R@4_std|R@8_mean|R@8_std|SR_mean|SR_std|SPL_mean|SPL_std|NE_mean|NE_std|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## End-to-End主要配对效应（seed平均）

|endpoint|split|control|metric|delta_pp|episode_ci95_pp|map_ci95_pp|map_equal_delta_pp|map_equal_ci95_pp|maps|
|---|---|---|---|---|---|---|---|---|---|


所有Seed、指标及对照的帮助/损害/不变数、逐地图效应和Leave-One-Map-Out结果见paired_summary.csv和paired_statistics.json。下表给出每个Seed的原始结果：

|endpoint|scope|representation|seed|split|samples|R@1|R@4|R@8|SR|SPL|OSR|NE|False_Stop_Rate|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## 几何复杂度与范围限制

geometry_subgroup_results.csv按预先固定的旋转最小包围矩形填充率0.85、长宽比3、最大地标面积1000m²、点状面积25m²及源类型/名称数分组；每组保留R@1/SR、NE和N，不剔除Polygon退化组。
多参照物的类型旗标可重叠，不能把组N相加当全样本数。关系组只使用明确解析且唯一绑定的公开参照，非GT目标–参照关系；不能强称所有复杂关系均得到验证。
dataset_group_counts.json显示Seen/Unseen“多条明确绑定关系”仅6/19个Query；Regular仅174/192，而Irregular为2177/2481，组间难度及大小等因素混杂，不能从不同组绝对SR差异直接推断因果交互。当前真实开放Polyline数量为0，Road-like结果仅说明TrafficRoad面状区域，不能外推到开放中心线。
完整分组、每图实值、paired_statistics.json中的help/hurt/unchanged、两种地图权重及Leave-One-Map-Out敏感性均随结果保存。百分位区间未作多重比较校正，分组分析属于探索性。

End-to-End三Seed平均分组结果（同一Query/Episode重复三个Seed不增加独立地图数；N为每Seed样本数）：

|endpoint|split|group|samples_per_seed|independent_maps|centroid_R@1_mean|bounding_box_R@1_mean|full_polygon_R@1_mean|centroid_SR_mean|bounding_box_SR_mean|full_polygon_SR_mean|full_minus_centroid_pp|full_minus_bounding_box_pp|
|---|---|---|---|---|---|---|---|---|---|---|---|---|


## 控制器与可复现性

全部使用同一个原始预训练Trajectory/Stop Head，SHA为51d8af7f05a9a718a9dc0ed12721a24ca6dd252e2714cb25948f22cb88cd2dbd，不微调；28×28、stop0.3、20决策、水平≤50m、垂直≤10m，原1mm预测目标到达门控保留。
重训后的候选上下文与旧head的分布可能不匹配，因此导航差异同时含表示和固定控制器迁移响应；不将其隐瞒或分别微调head以提高结果。
controller_static_weight_alignment.json直接比较参数张量：冻结C几何子模块与head训练时定位模型有70/80个张量不同（最大绝对差5.2411）；与其原始静态来源也有50/80个张量不同（最大绝对差0.00315）。因此连冻结Full基线也属于预注册旧head的迁移，不是正式完整PolyBelief的闭环复现。保留预先选定的head不变，不根据结果替换更有利的head；主要表示归因依据静态公平重训，导航只衡量固定head下的执行迁移。
本实验为几何/语言定位和控制器迁移，不启用RGB/历史，不把它与384或独立视觉残差的提升相加。
public_static_record只使用公开地标绑定/名称/轮廓，目标和未来轨迹仅在forward/rollout之后评分；渲染优化须先通过128条完整轨迹与原观测builder逐点一致性。
闭环CPU特征缓存按完整public_static_record内容及计划SHA精确键控，仅保存公开几何和虚拟零标签打包架子；不按GT目标或近似文本匹配。跨Seed复用不变输入，不复用其他权重的预测。单测验证缓存与原转换逐Tensor完全一致、各表示不修改共享输入；Seen128参考路径同时关闭缓存并使用原观测builder，以完整轨迹精确校验两项执行优化。
配置、源/权重SHA、Query清单、Train-Dev分组、初始化及每epoch数据顺序hash、训练断点、逐Query/episode预测和日志均原地保留。复现命令：`python -m scripts.run_polygon_ablation --resume`；统计重建：`python -m scripts.polygon_ablation_statistics`。
implementation_provenance.json及source_snapshot_manifest.json记录实现Commit与代码SHA；frozen_plan.json中的source_commit是实验开始前的基线Commit。旧架构args保留来源字段（如原continue_from/seed），实际独立随机初始化、Seed和预算以新config.json及顶层Checkpoint元数据为准，不执行旧continue_from。navigation_data_audit.json记录processed_citynav目录与revised静态记录的Query ID集合一致性及全部来源SHA，导航started.json另记录实际策略代码SHA。
静态评估沿用原static_recall的topk(8)并列排序，R@1/4/8由其前缀得到；闭环保留原模型topk(20)候选与原head选择。BF16并列分数时两者首位可能不同，不能把静态R@1和闭环SR的差额全部归因于执行控制。所有表示使用相同协议，不为个别表示修正并列顺序。

若主要End-to-End Unseen效应未获地图级正下界支持，论文核心创新应强调“使用完整公开几何的可解释接口/表示设计”，而不能声称已证明Polygon跨地图优于点/框。若仅Descriptor Only有效，也只能声称候选–地标描述符的条件增量价值。
