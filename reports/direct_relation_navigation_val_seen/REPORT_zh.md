# 原论文方案：直接地标关系指令的真实闭环成功率

## 结论

原论文GraphNav-V1在完整Val-Seen既有闭环记录中的直接几何关系任务：**536/1161成功，SR=46.17%**。同一轮完整2470条任务总体1206成功，SR=48.83%。这是真实闭环SR的事后分组，不是离线Goal R1，不是新模型或新训练成绩。

## 数据与协议核验

- 复用`Ablations/GraphNav/visual-residual-20261013/full_resolution_384/val_seen`中的完整predictions.jsonl、eval_started.json及metrics.json。目录名含旧AerialVG研究历史，但实际该轮`model=graphnav-v1`，visual control为real，未加载AerialVG视觉模型。
- 原方案为冻结GraphNav-V1 Stage1（原SigLIP视觉残差、因果质量历史及原Guarded-Blend），Checkpoint SHA256：`00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`。
- 使用该轮原协议及RGB配置，没有重新渲染、改变yaw或重新导航。**这是原论文历史协议结果，不是最近固定camera yaw=0的128条实验**，两者不能混为同一次评测。
- 逐条核对2470/2470：唯一索引、地图、目标/标注Episode标识及初始XYZ均匹配当前processed Val-Seen；重新汇总SR/OSR/SPL/NE/nDTW与原metrics一致。
- 分类直接调用既有TRAIN冻结规则（同spaCy/parser版本及无锚点专名处理），没有根据导航结果改规则。仅事后使用标注类别及显式名称身份核验，没有使用目标GT坐标构建关系或筛选地标。
- 不读取Test，不改模型或参数，不训练，不修改数据集和评测协议。此前建立的TRAIN筛选集与本次Val-Seen统计分开；本次仍完整覆盖全部Val-Seen，不移除困难任务后伪称总体提升。

## 全量分组

|关系类型|任务数|成功数|SR|OSR|SPL|NE(m)|nDTW|
|---|---:|---:|---:|---:|---:|---:|---:|
|直接几何关系|1161|536|46.17%|54.18%|0.4455|28.1956|0.3002|
|间接几何关系|771|391|50.71%|59.92%|0.4870|28.9708|0.3076|
|未找到明确关联|6|1|16.67%|33.33%|0.1656|58.5163|0.3186|
|分类不确定|532|278|52.26%|59.59%|0.5003|29.2029|0.3125|
|全部|2470|1206|48.83%|57.09%|0.4695|28.7282|0.3052|

本轮Val-Seen没有direct_identity样本，直接组即direct_geometry。各组互斥；存在直接及间接路径时沿用原分类最短可靠路径归为直接。

## 直接关系组的目标类别

|目标类别|任务数|成功数|SR|NE(m)|
|---|---:|---:|---:|---:|
|Building|698|287|41.12%|29.9936|
|Car|351|195|55.56%|23.8511|
|Ground|75|33|44.00%|32.3904|
|Parking|37|21|56.76%|26.9888|

## 解释与限制

直接组有625条失败，说明问题并不只存在于多跳关系指令。但不能仅凭该表断定失败来自语言、视觉或控制，也不能据直接组SR更低认定直接关系本身更难：该组60.1%是Building，类别和地图构成存在混杂。

分类为自动语义分析而非逐条人工金标准，“直接”也可能是on某条长道路或near某建筑，并不确保目标唯一、距离近或可见。未关联组只有6条，不作稳定泛化结论。

直接组覆盖23地图。简单Episode Wilson95%区间43.32%–49.04%；按地图分组bootstrap（seed0、2000次）区间约41.16%–50.09%。这些区间只反映样本/地图重采样波动，不包含自动分类误差、模型随机种子波动或新场景泛化保证。

## 复现与验证

`python -m scripts.report_direct_relation_navigation --annotations <processed_citynav/citynav_val_seen.json> --catalog <cityrefer_original/objects.json> --evaluation <existing/full_resolution_384/val_seen> --output <new-directory>`

`summary.json`保存全量及各组指标、原运行配置、模型SHA和数据/源码/预测文件SHA；`episodes.jsonl`保存全部2470条原指令、自动关系路径及原导航指标。相关11项测试通过，其中9项原解析测试、2项新增SR聚合/重复记录拒绝测试。
