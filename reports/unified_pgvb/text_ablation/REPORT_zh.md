# PR23 Unified V2精简文本：完整训练、诊断及Val-Seen闭环

完成：只新训练landmark_relation；relation_only已取消，full_text只复用旧Checkpoint。未合并main，未运行Test。

## 配置和数据口径

沿用原Unified V2初始化/架构/Geometry+CandidateAnchorRelation/28×28网格及10D候选—Polygon几何；seed42、5轮、batch8、Geometry1e-5/Relation1e-4、原CE+0.05KL。PGVB/BERT/控制器冻结，RGB关闭以隔离语言；原Stable Goal、Trajectory/Stop及阈值不改。

全21,878条TRAIN每轮均拟合；Birmingham7,619条是TRAIN内拟合诊断，不是独立Dev泛化。5轮13,675更新，耗时1099.34s。既有架构使用geometry节点投影路径，GAT消息传播未启用，与旧实验一致。原始权重保留。

公开名称只从原文匹配FULL CityRefer目录，同名合法片段全部保留；超16容量整行空Anchor回退，不据GT选择。统计：{'name_only': 8199, 'paired_words': 12458, 'no_public_anchor': 1221, 'invalid_public_polygon_shape': 532, 'all_public_polygons_exceed_capacity': 234}。精简JSONL直接替换instruction，BERT/Rule/关系字段/合成节点同步重算；地图节点仍保留地标名字。修复别名传递到Rule问题。词法邻接不等于Target作用域已验证，between等未覆盖结构可能丢失。

完整文本平均32.47 WordPiece、精简7.91（含CLS/SEP），max_length96不变；仅全文11条超过96。GT只用于损失和事后评分。

## Birmingham全部7619条

|同一公共Polygon绑定|R20|Top5/20m|Top15/20m|NE(m)|
|---|---:|---:|---:|---:|
|full_text|64.195%|81.992%|90.104%|33.8403|
|landmark_relation|32.734%|53.655%|71.282%|46.7420|

新模型对公共绑定全文：{'fixes': 355, 'breaks': 2752, 'net': -2397, 'changed_top1': 6178, 'mean_NE_improvement_m': -12.901694297790527}。

旧全文原生绑定R20=70.088%，NE=20.3788m；改公共绑定后64.195%，净破坏449条，确有输入分布变化，不能全归因删词。冻结全文直接使用精简文本R20=19.438%；训练适配回到32.734%，仍未恢复全文水平。没有重训匹配分布的全文控制，适配与文本改变不能完全分离。

新Relation分支对新Geometry纠正339、破坏233、净+106，最大残差1.856。分支有用不代表整体更好。TopK是世界网格位置召回，不是视觉实例识别。

## 手工Rule与网络

additive_off只取消直接加分；manual_off还清掉规则字段、关系节点和对应名字/连接，保留公共Polygon先验及网络。它们是冻结干预，不是另训模型。

|模型|干预|R20|NE(m)|
|---|---|---:|---:|
|full_text|additive_off|62.948%|34.6438|
|full_text|manual_off|63.657%|34.0881|
|landmark_relation|additive_off|32.222%|47.4195|
|landmark_relation|manual_off|32.734%|46.8887|

64个公开单Polygon地标，无GT选样，方向反事实如下：

|模型/干预|N→S工程方向响应率|E→W响应率|N区域质量响应率|E区域质量响应率|N−S质心差(m)|E−W差(m)|
|---|---:|---:|---:|---:|---:|---:|
|full_text/full|48.44%|56.25%|51.56%|51.56%|-0.601|1.081|
|full_text/additive_off|0.00%|0.00%|0.00%|0.00%|-20.104|-26.968|
|full_text/manual_off|31.25%|51.56%|32.81%|54.69%|-0.748|0.191|
|landmark_relation/full|34.38%|46.88%|29.69%|34.38%|-1.430|0.294|
|landmark_relation/additive_off|0.00%|0.00%|0.00%|1.56%|-26.736|-33.217|
|landmark_relation/manual_off|37.50%|51.56%|37.50%|54.69%|-0.487|-0.204|

64例原始反事实全部保留；其中NS/EW各1例的Anchor中心越过最外网格中心，某半平面无候选支持，不能视为关系判断错误。每轴63例可行子组另见counterfactual_support.json；不改变完整导航分母。north=+Y只是旧Rule工程约定；TIF没有CRS/地理配准，以上不是地理方向准确率，不据GT翻转轴。仅去掉直接加分后明显反向，显示规则特征学习与直接加分存在补偿/抵消。清除全部手工关系输入后，新模型方向响应未稳定吻合工程约定，不能据热图移动宣称学会语义。全文模型的合成简句也存在输入分布变化。

## 信息损失证据

Birmingham全文有7180种严格输入签名，精简仅1240种。精简后322组相同输入对应GT跨度>40m，涉及6322条；全文对应仅2条。这是事后标签冲突分析，不是GT选Goal或准确率上界，也不说明模型已最优或6322条都必失败。相同输入无法同时区分这些目标，更多Epoch不能恢复被删的身份、距离和复杂约束信息。

## 完整2470条Val-Seen真实闭环

最终5轮Checkpoint及代码先冻结，见FROZEN_BEFORE_VAL.json，无Val调参。两组同任务、起点、公共Polygon和控制器；原目标GT只在rollout后评分，未以地标Polygon距离替换原20m目标中心指标。全部失败和无候选计入分母。

|方法|N|SR|OSR|SPL|NE(m)|nDTW|决策ms/步|
|---|---:|---:|---:|---:|---:|---:|---:|
|full_text|2470|45.749%|53.279%|0.4440|37.9596|0.2888|24.213|
|landmark_relation|2470|34.089%|41.538%|0.3297|41.4197|0.2766|23.714|

配对纠正166、破坏454、净-288。

停止/Goal事后归因：{'full_text': {'goal_wrong_navigation_failed': 1340, 'OSR_success_SR_failed': 186, 'goal_jumps_over20m': 0, 'correct_policy_stop': 1130, 'policy_stop': 2470}, 'landmark_relation': {'goal_wrong_navigation_failed': 1628, 'OSR_success_SR_failed': 184, 'goal_jumps_over20m': 0, 'correct_policy_stop': 842, 'policy_stop': 2470}}。这是预测Goal与GT的事后对照，GT未进入Stop或导航。官方物理目标分组bootstrap的SR差95%区间[-14.484,-8.951]个百分点。单seed，不宣称多种子稳定性。策略决策时延排除环境渲染，共享GPU，不是独占硬件成绩。

## 决策与交付

本轮完整Val-Seen不支持用该精简方案替换全文。 目标身份和约束丢失已经由标签冲突检查证实；后续优先保留这些必要信息、独立核对方向参考系和关系作用域。本轮不增加网络、重训全文或调Rule/Stop。

381项CPU测试通过，见CPU_tests.txt。GPU训练Smoke、4次TRAIN导航接口通过，真实公共记录与准备文件逐项一致。adversarial-review服务超时，未列为独立审查通过。两组均2470次Policy Stop、Goal大幅跳变0，正确Goal却导航失败0；本轮失败指向目标定位，而非Stop未触发。最终代码/权重SHA已再次核验。

最终Checkpoint：/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v2_text_ablation_s42_20261105/train_landmark_relation/last.pt，SHA256 0c441e20922c3e75914022cf85bf0131917c4ea7f57d964e09401bf80540cc1d。

服务器根目录：/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_v2_text_ablation_s42_20261105，保存模型、日志、全部轨迹/逐步RGB SHA/Belief及预测。GitHub保存汇总、逐任务CSV、真实热图和轨迹，[HTML图册](index.html)。

反事实边界检查：原始64个地标全部保留；按Anchor与28×28网格中心检查，南北两侧均有候选支持的为63个，东西为61个。不可行半平面不能解释为模型方向错误；可行子集的质心/区域质量响应单独保存于counterfactual_support.json。上述64例原始响应表不作地理方向准确率结论。
