# Geometry-Guided Visual Belief Learning：第一阶段计划

用户授权先训练地标Visual Belief，再决定是否进入直接关系目标推理。独立分支实现；原GraphNav-V1 Checkpoint、几何分支、默认推理及控制器不变，无AerialVG、无Test。

## 数据与已完成审计

后续训练必须使用已生成的地标关联TRAIN子集，排除原自动no_confirmed_anchor/uncertain。只读取其公开名称、原文及既有地图分组，不读取旧任务目标坐标/Contour作本阶段输入。原SigLIP缓存筛选后，fit为1112个有正负弱区域窗口的样本/297名称；dev仅19样本/6名称；confirm31样本/9名称且全是Road，评价覆盖不足。

补充公开地标训练视图：仅从筛选TRAIN原文实际引用的具名地标中按seed0与地图分层选择，计划fit各75个Building/Road、dev和confirm各16个Building/Road。每个名称在公开Polygon代表点附近4个位姿采样，水平偏移25m、高度50/80m、固定camera yaw=0，合法地图边界裁剪。真实正射RGB384px及深度有效性，不读原CityNav最终目标点。它们是离线训练位姿，不是人类轨迹，也不是实际闭环成绩；不能用于在线未来视图输入。

Polygon/Footprint仅提供保守几何弱区域标签：有效1patch窗口、内缩0.5m覆盖≥90%为正，外扩0.5m覆盖≤1%为负，边界/无效为未知。多碎片同名Road保留并集；Building/Road跨类型重名不强制监督。所有可见性仍为unknown_geometry_proxy_only，不冒充人工像素GT。

## 网络与损失设计

冻结原几何A3与SigLIP2主干。为每个名称重新计算原几何分支Belief和名称/类型文本向量，不能复用旧目标指令的条件缓存。新增轻量视觉评分头，学习当前图像特征、名称和公开类型描述的匹配；不向该可训练评分头提供Polygon绝对坐标、像素Mask、UAV绝对XY或teacher logits。

Polygon用于确定观察支持、弱监督和最终融合的几何先验；Footprint/UAV用于把视觉证据投到原28×28世界网格，不进入视觉评分捷径。保留原生SigLIP共享空间分数作为初始化基线。

三项训练损失：观察支持域内的几何Belief蒸馏KL、保守地标区域定位损失、多视角相同公开地标/不同地标对比损失。先真实GPU smoke及恢复测试，再固定seed0、12轮、batch16、AdamW lr1e-3，不在正式Val调参。原几何分支及SigLIP不训练。

需分别报告：G（几何）、原生视觉V、学习视觉V及G+V。原始V只表示当前可观测域中的视觉位置分布，不能把它当成已搜索完地图的全局概率；G+V用观测支持内的似然更新，未观测域无负证据。无可用几何观察支持时保留G，不构造“目标不存在”的Null标签。

## 反事实与门槛

保持地图、样本、位姿与监督不变，冻结后测错误RGB、有效位置打乱、同类不同地标RGB替换；另用同类公开Polygon互换重新生成几何teacher，查询名称和RGB保持不变，不能将原始V对Polygon不敏感直接称为识别正确。所有错配规则只依公共信息与seed0，不用目标GT位置。

跨视角按相同公开名称组统计，一组多帧不能当独立目标。内部dev/confirm仍保留历史使用状态，不重新冒充未见确认数据。

第一阶段至少需要：开发组弱区域Top1相对原生视觉及错配RGB有净改善，G+V纠正≥误改且NE不劣于G；还需公开全部反事实及观察支持覆盖，不能只用和几何一致来宣称视觉有效。若不满足，不进入直接目标关系训练，不为追求通过扩大模型/反复改阈值。

通过后冻结配置进行64条地标闭环回归，同一控制器、同一观察yaw/分辨率，几何基线与新方法配对。TRAIN缓存为固定yaw0，因此回归需对两组统一明确该采图设置，不能直接把此前原生yaw的95.31%当配对基线。随后才用TRAIN直接关系任务训练最终目标Visual Residual；目标属性、参照及关系分开，原GT仅用于TRAIN损失和事后评价。

## 执行与恢复

缓存/权重/完整观测保留服务器`GraphNav/runs/geometry_guided_visual_belief/`；代码、源SHA、统计、中文报告及代表性PNG提交Git。数据采集tmux=geometry-guided-views。资源预检技能的GPU扫描误报无GPU，以实际nvidia-smi和CUDA检测为准，现有其它任务不停止。
