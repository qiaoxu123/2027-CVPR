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
