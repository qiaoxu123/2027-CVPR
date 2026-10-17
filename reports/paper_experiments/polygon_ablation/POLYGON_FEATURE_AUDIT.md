# PolyBelief Polygon Geometry 信息通路审计

初版审计和frozen_plan.json在任何新实验结果产生之前写入。后续仅追加消费范围及公开分组数量核验，不变更表示、预算或分组规则；版本记录见Git历史及源快照。原权重、共享数据和归档输出不修改。

|通路|使用位置|包含的完整几何信息|本实验处理|
|---|---|---|---|
|候选–地标10-D|candidate_geometry.relative_entity_features → geometry_proj / pair attention|相对XY、中心距离、方向、signed boundary、Inside、AABB宽高、开放线类型|所有组相同10维；不可用值乘显式mask置零；不新增可学习参数|
|节点10-D|semantic_graph._physical_node_geometry → gnn.node_geometry，即使variant=geometry仍使用|中心、宽高、周长、面积、方向、面标志、顶点数|Descriptor Only完全保留；End-to-End重建各表示可用统计，Centroid移除全部形状量；BBox面积/周长/顶点数只从矩形推导|
|prior/prior_buildings|StaticLocalizationDataset._prepare → baseline、token_head、geometry_head|候选到完整区域的距离，内部为0|Descriptor Only原样；End-to-End为点/真实AABB区域/完整区域距离；building-only旁路也同步处理|
|rules|prior_and_rule_fields → baseline及评分头|near/along使用完整区域距离；direction/between使用顶点均值|Descriptor Only原样；End-to-End按允许几何重算，相同规则公式和sigma|
|relation_fields|rule_map → query → token attention/geometry head|上述单条规则场，可能绕过pair_geometry|同rules同步重算，不能只清除pair特征|
|图节点/边/拓扑|prepare_semantic_graph / TypedGeometryGAT|边界距离、相交、邻近、道路拓扑|所有组统一variant=geometry，无GAT消息；传入的edges/adjacency同时清零；不变更网络参数结构|
|候选位置编码|FullTokenGeometryLocalizer._candidate_positional_features|公共地图网格；可选参照中心|统一原28×28与同一relative_position配置；中心约定完全相同|
|公开地标绑定/选择|public_static_record、数据转换、静态打包|名称/类型/实例/原顶点均值；可选道路选择|所有组相同名称类型/实例顺序/有效mask，禁止GT最近地标；当前正式数据max_roads=0|
|训练标签|cross_entropy_goal；评测层|目标坐标只用于loss与forward之后的指标|不进入geometry_view、_batch_inputs或模型forward；Train-Dev组划分仅使用Train对象ID|
|RGB/历史/完整观测|正式完整模型其他模块|可能额外包含形状或目标外观|本次静态几何模型及闭环均不启用RGB/历史；不能冒充完整RGB方法的同配置收益|
|固定Trajectory/Stop|A3TrajectoryPolicy.predict_field / 原Guarded-Blend|候选上下文、候选坐标、公开姿态和地图范围|全组使用相同预训练head SHA，不重新微调；重训上下文分布改变是明确的控制器迁移局限|

中心点遵循原代码的顶点算术均值，非面积质心，也未剔除重复闭合顶点。所有组共享这个原始点。BBox另保留真实轴对齐包围盒；此基线实际为“共享原始中心点 + AABB”，以避免中心定义差异混淆形状贡献。

开放Road/Polyline不闭合；Full使用真实开放线距离且Inside恒0。Centroid与BBox缺少真实线形时，boundary/inside通道mask为0，near/along仅用可用中心点，绝不伪造连线。TrafficRoad/Footpath等CityRefer面状轮廓继续按其源类型作为区域，不混作中心线。

## 两类实验的归因边界

冻结干预使用正式冻结C Epoch1中现有几何定位子模型，控制其权重不变。Descriptor Only只替换pair_geometry，其prior/rules/relation_fields及节点形状仍保留完整Polygon，不能称Centroid完整表示。End-to-End同时替换全部可学习模型实际消费的几何通路。

No Inside仅清除Inside通道，signed boundary的负号仍编码Inside；因此额外加入No Inside Information，把边界距离取绝对值并清除Inside。No Boundary/No Extent也只是冻结模型的通道依赖诊断，其他旁路仍在，不是公平表示比较。

独立重训保持相同原网络、15epoch、batch16、lr5e-5、AdamW、损失及随机种子0/1/2，匹配初始化与数据顺序；只用Train内部固定对象分组Dev选择checkpoint。Descriptor Only与End-to-End分别报告；Full输入相同的两scope按同一SHA复用，避免无意义重复训练。

几何分组阈值在frozen_plan.json中预先固定，不读取结果；按公开参照物形状统计，不使用GT目标形状或不可靠目标–参照物标签。多地标的类型分组允许重叠并分别报告N。4张Unseen地图不能当2679或2697个独立场景。

新增三项预结果冻结旁路干预：仅把prior、仅把rules+relation_fields、仅把节点形状改为Centroid允许信息，其他模型输入及权重不变，单独量化旁路依赖。它们同样不是公平重训表示比较。

补充消费范围核验：`prior_buildings`只由`variant=building_prior`读取，在本次`variant=geometry`中不参与logits；仍与prior同步重建，以免将来复用其他variant时产生旁路。关系节点也不在`entity_valid`的候选–地标注意力有效集合中；其显式约束主要通过rules/relation_fields进入。本表“保留/重建”不意味着每个打包字段都在当前variant中产生影响。

公开分组审计（dataset_group_counts.json）：当前Train/Seen/Unseen实际开放Polyline引用均为0；Road-like结果来自源TrafficRoad面状区域，不能作为开放道路中心线优势证据。Seen/Unseen仅6/19个Query具有至少两条明确且唯一绑定的解析关系，复杂关系组统计只能作为小样本探索。Seen 23图、Unseen 4图；regular/irregular组显著不平衡，报告中必须同时保留N。

控制器来源补充核验（controller_static_weight_alignment.json）：虽然预注册head的来源SHA严格验证通过，其训练定位模型与冻结C的几何子模块有70/80个张量不同（最大差5.2411）。这意味着包括冻结Full在内的闭环均是同一旧head的迁移测试，不是正式完整模型的导航复现。该head在任何本轮结果前已固定，不因差异而改选；公平表示贡献主要以静态重训验证，导航迁移限制单独报告。当前冻结C几何参数与initial_sources中原始静态来源也有小幅差异，不能把initial_sources的静态SHA当成本次实际冻结参数；实际来源始终是BASE C Epoch1权重SHA。

No Inside Information仅指10-D候选–地标描述符中的Inside通道及signed boundary符号，prior等其他Polygon旁路在Descriptor Only中仍保留，不能据此宣称模型完全没有Inside相关信息。
