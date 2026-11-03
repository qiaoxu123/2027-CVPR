# Changelog

## [landmark-chain-pilot] - 2026-11-03
### Features
- 独立固定yaw观测适配器及TRAIN分层采集、冻结SigLIP2特征缓存脚本。
### Design Rationale
- 明确区分相机yaw与飞行器yaw，并让Footprint与真实采样一致；不修改原导航默认行为。
- 图像与公开输入独立于目标监督标签保存；先验证原始视觉分支再训练关系链模块。
### Notes & Caveats
- 本阶段24实例仅用于机制开发，不是导航或泛化成绩；参考轨迹采集不等于闭环。
- 语义角色由当前会话AI整理，非人工金标准；保留原指令及参考系不确定性。

## [local-chain-position-pilot] - 2026-11-04
### Features
- 新增独立局部位置分布头、144个当前观测候选、公开地标几何特征及可恢复TRAIN训练入口。
- 完成400物理目标固定yaw采集、地图分组和一次独立确认，保留全部消融与负结果。
### Design Rationale
- 允许观测内位置假设与原全局位置竞争，解除原有界视觉修正限制；不修改原主权重或控制器。
### Notes & Caveats
- 独立Goal位置有提升，但Shuffle没有退化、角色查询没有净增益，尚未证明视觉实例对应有效；没有新闭环SR。
- 监督为弱世界位置，不是人工像素实例标注。原导航默认入口保持不变。

## [local-chain-closed-loop-diagnostic] - 2026-11-04
### Features
- 新增独立冻结TRAIN闭环入口，复用原Trajectory/Stop与控制器；支持原完整RGB历史及局部头Real/Zero/Shuffle/A3对照。
- 完成24任务×5组实际闭环、失败/Stop归因和6例真实决策位姿RGB图册。
### Design Rationale
- 用真实闭环检验离线位置增益，避免把参考轨迹回放或Goal R1当SR。
### Notes & Caveats
- Real SR15/24、Original13/24，但Shuffle同为15/24；视觉空间对应未证明。
- 新头无Stop且SPL下降，需要保持固定世界Belief支持；原入口、权重、Stop阈值不改。无Val/Test成绩。

## [fixed-world-belief-interface] - 2026-11-04
### Features
- 局部位置概率可微投回原固定世界网格，默认使用grid；Direct只作负对照保留。
- 逐决策记录真实观测SHA、相机姿态和Footprint，并核验实例重建图像内容。
### Design Rationale
- 相机锚定patch网格不含光心，无法满足严格到达Guard；固定世界Goal支持修复可到达性，不改Stop阈值。
### Notes & Caveats
- 同25条TRAIN开发任务，Direct→Grid Stop2→25、SR18→20，但原完整模型21成功；新方案尚未超过基线。
- 开发结果不是独立确认或Val成绩；32项测试通过，原权重/普通入口保持不变。

## [local-chain-onpolicy-data] - 2026-11-04
### Features
- 新增TRAIN拟合组真实闭环观测导出，逐帧匹配运行时RGB像素SHA，公开观测与私有监督分离。
- 特征缓存可沿用冻结地图划分；训练可加入同一拟合目标的新观测，拒绝开发目标和重复帧。
### Design Rationale
- 用实际访问观测检验参考轨迹缓存的分布差异，保持同架构、优化更新次数及控制协议。
### Notes & Caveats
- 世界20m支持不是像素实例或可见性标签。新实验收益必须由真实闭环衡量，不作为正式Val/Test结果。

## [single-world-likelihood-pilot] - 2026-11-04
### Features
- 新增独立V3观测似然头：在同一世界网格加入局部观测能量，不再将全局先验的局部副本作为第二组概率质量。
- 增加零证据保持先验、重复局部采样不放大似然、图外点不污染边界及真实梯度测试。
### Design Rationale
- 中性观测不应仅因局部候选采样集中而改变全局目标；区分区域的采样密度与新增证据。
### Notes & Caveats
- 原V1/V2权重、默认入口和控制器均保留。合成反例不是实际发生比例；新头需重新训练并以闭环判断。

## [deployment-envelope-audit] - 2026-11-04
### Features
- 新增公开位姿高度范围筛选与同优化更新预算采样对照；原始起点观测保留。
- 新增冻结SigLIP2当前RGB裁剪编码审查及24案例浏览图册；新增独立确认物理组冻结工具。
### Design Rationale
- 先对齐参考轨迹训练观测与部署动作范围，避免在不可部署超低空帧上推断视觉能力；裁剪不冒充新增原始细节。
### Notes & Caveats
- 高度过滤开发SR88%仅比Original多1条，独立确认尚待完成。高度范围不等于完整可达性或可见性，原模型与控制协议保持不变。
