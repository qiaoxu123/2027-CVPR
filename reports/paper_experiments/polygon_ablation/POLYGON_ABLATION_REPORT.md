# PolyBelief Polygon Geometry 独立贡献验证

状态：部分完成；未完成项目保留Pending。[信息通路审计](POLYGON_FEATURE_AUDIT.md)与frozen_plan.json先于实验结果产生。
主对照是End-to-End独立重训；Descriptor Only仅解释描述符增量，冻结干预仅解释已训练权重的依赖。
相同10-D/模型容量、3个种子、15epoch、batch16、AdamW5e-5。训练样本按Train对象分组保留固定Dev，选择只用Dev R@1及NE；未用Unseen/Test选模或修改配置。
Full Polygon在两个口径的输入完全相同，每seed按同一SHA复用；共18个逻辑条件、15次独立训练。seed0预先指定优先闭环，并保留seed1/2所有结果。
所有静态评估完成标志all_static_complete.json先于任何闭环。静态Seen2426/Unseen2679，导航Seen2470/Unseen2697；原始定位/导航结果未冒充本实验结果。

## 三个最终问题

1. Full Polygon是否比Centroid/Bounding Box更好：以下End-to-End重训配对效应给出实际方向，不预设正收益；统计区间跨零或点估计为负时，不能写成普遍优势。
2. Seen/Unseen是否一致及统计支持：分别报告；4张Unseen地图仅为4个独立地图簇，主要依据整地图cluster CI和Leave-One-Map-Out，不能以Episode CI替代。
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


## 三Seed均值及样本标准差

比例列均为0–1，NE为米；delta_pp及map_ci95_pp明确使用百分点。每seed原始结果分别保留在两个results.csv中，不选择最佳seed。

|endpoint|scope|representation|split|seeds|samples_per_seed|R@1_mean|R@1_std|R@4_mean|R@4_std|R@8_mean|R@8_std|SR_mean|SR_std|SPL_mean|SPL_std|NE_mean|NE_std|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## End-to-End主要配对效应（seed平均）

|endpoint|split|control|metric|delta_pp|map_ci95_pp|maps|
|---|---|---|---|---|---|---|

## 几何复杂度与范围限制

geometry_subgroup_results.csv按预先固定的旋转最小包围矩形填充率0.85、长宽比3、最大地标面积1000m²、点状面积25m²及源类型/名称数分组；每组保留R@1/SR、NE和N，不剔除Polygon退化组。
多参照物的类型旗标可重叠，不能把组N相加当全样本数。关系组只使用明确解析且唯一绑定的公开参照，非GT目标–参照关系；不能强称所有复杂关系均得到验证。
完整分组、每图实值、paired_statistics.json中的help/hurt/unchanged、两种地图权重及Leave-One-Map-Out敏感性均随结果保存。百分位区间未作多重比较校正，分组分析属于探索性。

## 控制器与可复现性

全部使用同一个原始预训练Trajectory/Stop Head，SHA为51d8af7f05a9a718a9dc0ed12721a24ca6dd252e2714cb25948f22cb88cd2dbd，不微调；28×28、stop0.3、20决策、水平≤50m、垂直≤10m，原1mm预测目标到达门控保留。
重训后的候选上下文与旧head的分布可能不匹配，因此导航差异同时含表示和固定控制器迁移响应；不将其隐瞒或分别微调head以提高结果。
本实验为几何/语言定位和控制器迁移，不启用RGB/历史，不把它与384或独立视觉残差的提升相加。
public_static_record只使用公开地标绑定/名称/轮廓，目标和未来轨迹仅在forward/rollout之后评分；渲染优化须先通过128条完整轨迹与原观测builder逐点一致性。
配置、源/权重SHA、Query清单、Train-Dev分组、初始化及每epoch数据顺序hash、训练断点、逐Query/episode预测和日志均原地保留。复现命令：`python -m scripts.run_polygon_ablation --resume`；统计重建：`python -m scripts.polygon_ablation_statistics`。

若主要End-to-End Unseen效应未获地图级正下界支持，论文核心创新应强调“使用完整公开几何的可解释接口/表示设计”，而不能声称已证明Polygon跨地图优于点/框。若仅Descriptor Only有效，也只能声称候选–地标描述符的条件增量价值。
