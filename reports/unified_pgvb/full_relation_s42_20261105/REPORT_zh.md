# UnifiedPGVB完整TRAIN联合训练与完整Val-Seen闭环

仅seed42。所有模型及参数在Val评分前冻结，不用Val选择Checkpoint或调整参数；Test未访问。

## 完整闭环
|方法|N|SR|OSR|SPL|NE(m)|nDTW|时延ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|Original Geometry|2470|47.77%|55.51%|0.4633|29.198|0.3029|11.33|
|Trained Geometry|2470|50.24%|57.33%|0.4875|28.465|0.3051|11.09|
|Geometry+Relation|2470|50.24%|57.33%|0.4875|28.463|0.3051|24.64|
|Geometry+Relation+RGB|2470|50.00%|57.41%|0.4848|28.522|0.3050|34.99|

原始Geometry是额外完整基线；其余三组共享训练后的Geometry。Relation与RGB的Geometry/Relation权重逐张量相同，只有PGVB阶段权重变化。全部2470任务计入，空参考和推理异常不删除。

## 逐任务配对与不确定性
```json
{
  "original_geometry->geometry": {
    "correct": 123,
    "damage": 62,
    "net": 61,
    "SR_delta": 0.024696356275303644,
    "map_bootstrap_95CI": [
      0.008335410844339081,
      0.04354870949014414
    ]
  },
  "geometry->relation": {
    "correct": 0,
    "damage": 0,
    "net": 0,
    "SR_delta": 0.0,
    "map_bootstrap_95CI": [
      0.0,
      0.0
    ]
  },
  "relation->rgb": {
    "correct": 2,
    "damage": 8,
    "net": -6,
    "SR_delta": -0.0024291497975708503,
    "map_bootstrap_95CI": [
      -0.005374203382824072,
      0.0
    ]
  },
  "original_geometry->rgb": {
    "correct": 122,
    "damage": 67,
    "net": 55,
    "SR_delta": 0.022267206477732792,
    "map_bootstrap_95CI": [
      0.006085854208199705,
      0.04077020400405318
    ]
  },
  "geometry->rgb": {
    "correct": 2,
    "damage": 8,
    "net": -6,
    "SR_delta": -0.0024291497975708503,
    "map_bootstrap_95CI": [
      -0.005374203382824072,
      0.0
    ]
  }
}
```
单随机种子不能证明训练随机性稳健；地图分组bootstrap仅描述这批评测任务的采样不确定性，不等于多种子显著性。

## 训练范围
Phase1：完整Revised TRAIN 21878条×5轮，0转换跳过。Geometry和AerialVG-inspired Relation联合训练。不是官方AerialVG预训练参数。BERT/SigLIP2冻结，既有PGVB在Phase1冻结；不是全视觉骨干端到端训练。
Phase2：249帧、205条fit指令的真实当前RGB缓存训练PGVB三轮，Geometry与Relation冻结。支持局部Anchor的筛选与运行时一致，不制造画外视觉证据；此子集较小并偏向直接关系，不宣称全样本视觉训练。
原始Stop概率阈值0.3、到达守卫1mm、StableGoal margin0.35/连续2帧、20步Horizon及20m成功半径均不变。

## 失败分析
互斥主分类仅按轨迹/Goal可证事实：推理错误、从未有正确Goal、曾正确Goal但未到达、进入20m后未最终成功。它们不能单凭NE区分像素实例错误、错误Polygon绑定或遮挡；无独立标注时原因未知。
详细分类与各分层在failure_analysis.json、metrics.json；保存了首次进入20m时Goal、Stop概率、到达守卫及高度守卫，不自动修改Stop。
Belief 20m纠正/破坏是位置指标，不等于真实实例匹配准确率；公开Polygon是地图弱监督，不是人工像素GT。

## 决策
Geometry→Relation净纠正+0条；Relation→RGB净纠正-6条。完整负结果同样保留，不根据这次Val结果追加参数搜索。
公开Polygon与语言作用域、RGB确认是否真能识别目标、观察不足及原控制器Goal保持均需分别看实际轨迹；不能从Relation残差非零就宣布理解关系。

## 复现与图像说明
Checkpoint路径与SHA256见checkpoint_SHA256.json；冻结协议见freeze_*.json；CPU与GPU Smoke证据均随报告保存。所有案例RGB由实际轨迹位姿重渲染并逐帧SHA校验与在线真实观测一致，384×384原始观测，未冒充高清。地图轴为工程World XY，不声称其已完成地理北向配准。
所有图包含成功、破坏及剩余失败；红叉为事后GT目标位置、青色为公开Polygon、白线为实际预测轨迹。

## 分层完整结果

直接/间接为自动语言路径分组，非人工关系真值；另列历史严格56条子集。各分母不混用。

|分组|方法|N|SR|OSR|NE|
|---|---|---:|---:|---:|---:|
|direct|Original Geometry|1309|46.29%|53.40%|28.288|
|indirect|Original Geometry|741|51.82%|59.51%|28.914|
|unresolved_or_no_relation|Original Geometry|420|45.24%|55.00%|32.536|
|previous_strict_direct_56|Original Geometry|56|50.00%|51.79%|24.008|
|direct|Trained Geometry|1309|49.12%|55.84%|27.599|
|indirect|Trained Geometry|741|52.90%|59.92%|28.374|
|unresolved_or_no_relation|Trained Geometry|420|49.05%|57.38%|31.326|
|previous_strict_direct_56|Trained Geometry|56|60.71%|62.50%|23.240|
|direct|Geometry+Relation|1309|49.12%|55.84%|27.595|
|indirect|Geometry+Relation|741|52.90%|59.92%|28.374|
|unresolved_or_no_relation|Geometry+Relation|420|49.05%|57.38%|31.326|
|previous_strict_direct_56|Geometry+Relation|56|60.71%|62.50%|23.240|
|direct|Geometry+Relation+RGB|1309|49.05%|55.92%|27.616|
|indirect|Geometry+Relation+RGB|741|52.50%|60.05%|28.464|
|unresolved_or_no_relation|Geometry+Relation+RGB|420|48.57%|57.38%|31.446|
|previous_strict_direct_56|Geometry+Relation+RGB|56|60.71%|62.50%|23.240|

## 失败与实际控制

|方法|从未正确Goal|曾正确Goal但未进入|进入后未最终成功|推理错误|正确PolicyStop|重规划|RGB更新帧|
|---|---:|---:|---:|---:|---:|---:|---:|
|Original Geometry|1099|0|191|0|1180.0|0|0|
|Trained Geometry|1054|0|175|0|1241.0|0|0|
|Geometry+Relation|1054|0|175|0|1241.0|0|0|
|Geometry+Relation+RGB|1052|0|183|0|1235.0|37|2129|

错误Goal处停止不能自动归因为Stop网络失败；需结合首次20m进入时的Goal与守卫证据。没有可靠像素实例标注，Polygon/RGB错配或真实目标不可见的原因保留未知。

## 关键归因：Relation与RGB

Relation的SR纠正/破坏为0/0；2470条中只改变1条初始Top-1，几何Top-2 logit间隔中位数0.11413，而Relation最大修正中位数仅0.00423。按逐帧概率计算，平均Belief质心变化约0.00194m，不能把非零梯度/残差称为有效关系推理。

RGB纠正2条、破坏8条，净-6。8条破坏全部曾进入20m且都出现视觉触发重规划（690、999、1016、1037、1210、1213、1766、2089）。并非控制器不会到达正确Goal：它们最终都在新的错误Goal处Policy Stop。没有Horizon推延或阈值修改。

例如任务690：原Goal到GT约12.1m；第5步到原Goal只剩0.088m，Stop概率0.386，但仍未满足原1mm到达守卫。第6步视觉确认后重规划到GT约28m的错误Goal，最终在该点正常Stop。公开ROI的视觉AUC接近1只证明其地图区域可区分，不证明白车目标身份正确。该案例不能作为放宽Stop的依据。

结构限制：现有PGVB以地标名称/类别/轮廓评分，缺少目标外观与实例条件；其地标证据直接叠加到Target分布，Relation空间变换又接近零。角色不匹配是有代码与轨迹支持的解释，但没有独立像素GT，不能把全部错误进一步判为真实实例识别错误或Polygon绑定错误。

**下一步优先**：保留训练后Geometry作为本轮候选；先在TRAIN冻结Geometry，验证Relation能否通过受控关系监督形成有效空间残差，并检查PGVB的目标条件与可观测监督。不要继续加Attention层、增加Epoch或调Stop阈值；没有新增分支的独立收益前不替换main。

统计范围：Geometry本轮提升+2.47pp，地图块bootstrap 95%区间约[+0.83,+4.35]pp；仅seed42，不能替代多种子或Val-Unseen泛化验证。本轮没有区分普通Geometry继续训练与联合训练对Geometry的间接作用，需要另外的TRAIN训练对照才能归因。

## RGB回退与性能口径

Val-Seen所有任务均进入全语言模型，没有使用573/56严格解析清单筛选；直接/间接标签仅作事后分层。1748条没有唯一合法视觉Anchor，继续Geometry+Relation；完整RGB组共有1923条没有启用视觉更新、547条至少启用一次（2129个决策帧、37次重规划）。两项回退统计重叠，不相加；对应SR/NE见RGB_fallback_strata.json。无有效实体参考的分层在本次Val数据中为0条；TRAIN中91条无公开引用仍已训练。

PGVB训练排除日志为81个查询级歧义条目、986个帧级无支持条目，不能将1067日志条目都称为图像。249帧中GT位于Footprint的163帧，仅为几何代理，不是真实可见性。

BERT继承96-token上限，TRAIN只有11/21878条超过此长度；不能把本轮瓶颈简单归因于普遍文本截断。Geometry模块595341参数，Relation269666参数；Geometry的非当前geometry路径GAT层未获得梯度，列于TRAIN_module_audit.json。旧Direction/Joint/Appearance原型头未启用，不冒充完成训练。

时延为rollout的决策计时，含语言/Belief/控制器和必要视觉编码，不含环境RGB渲染；GPU为共享服务器，资源CSV为整卡指标，PyTorch峰值为本任务实际分配。各方法的真实闭环轨迹会改变观测，图中为各自最后真实观测的Belief，统一世界范围与色标；原生RGB逐帧SHA已校验。

本轮仅验证Val-Seen，不推断Val-Unseen/Test表现。独立adversarial-review调用因工具workspace routing失败未取得结果，不计为独立审查通过。

## 交付验证

364项CPU测试通过；12条案例均来自真实闭环，原生RGB与在线帧SHA一致。HTML及全部64个本地资产通过HTTP200检查，未宣称浏览器交互测试。四份完整轨迹压缩包均已逐行核验2470条、索引唯一。全部原始Belief NPZ保存在服务器实验目录，不提交约1GB缓存。

## 总体瓶颈与保留建议

RGB完整模型的1235条失败中，1052条从未选择到20m正确Goal，183条曾进入20m后又离开；没有“选到正确Goal却无法进入”的任务，也没有推理错误。四组均Policy Stop结束，没有靠增加Horizon提高SR。整体瓶颈首先是目标位置Belief/证据，而非控制器不会到达Goal。RGB下有29条最终失败轨迹曾出现正确原始Belief，这是可进一步审查的证据，但不是已验证可恢复的29条SR。

本轮最佳候选为训练后Geometry：50.24% SR、28.4654m NE；完整RGB为50.00% SR、28.5215m NE，不能为了方法复杂而优先采用后者。保留全部结果与原Checkpoint，暂不合并main；后续只在独立TRAIN验证可靠关系/目标证据，不在本批Val反复调参。
