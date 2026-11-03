# Active Goal：原始GeomBelief的地标关系链视觉分支

用户Goal：基于地标→中间区域/对象→目标关系链设计并验证原始视觉分支；不使用AerialVG。固定yaw采集；以真实闭环SR为准，冻结方案后先128条Val-Seen，SR严格超过65%（至少84/128）且有配对收益才扩大完整评测。可以重训，不限残差形式。

## 已完成
- 原始工程定位：/home/20261024_1/Workspace/2027-CVPR/GraphNav；原始论文GeomBelief/GraphNav-V1，SigLIP2而非AerialVG。
- 主权重runs/graphnav_v1_stage1/best.pt，SHA00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a。标准论文28×28候选、384px观测；root MAIN_SCHEME曾写384×384稠密候选，这是另一个推理设定，必须先实际核对，不能混用。
- TRAIN22,002指令/21,303唯一map+text、24图分析完成，分支research/instruction-landmark-association-20261103，Commit2303ce122762d5c977ffe88ad03f1b7fb4a8357e已推送旧仓库。
- 完整自动初筛：直接几何9356、自名称1、间接7182、无明确锚点325、未判明5138。公开名称可绑定20959（95.26%）。本会话101条分层语义复核校正估计：直接50.79%、间接45.70%、无明确锚点1.18%、不确定2.33%；不是人类金标准，直接/间接区间约±8pp，不可当精确全量真值。
- 车辆自动解析间接5205>直接3144，建筑直接5184>间接1544。需要区域/参照实例表示，不能把地标中心当目标。语言频率不证明SR因果收益。
- 旧主工作树有3个修改与恢复脚本等未提交工件，保持不动；新Goal在此独立worktree/分支实施。

## 当前未完成
- 尚无新视觉模块设计定案、训练或导航结果；不要声称新SR。
- 需先读取graphnav/stage1.py、stage1_runtime.py、原SigLIP2 ROI/记忆实现与已有CVR/局部语义负结果，确认瓶颈。
- 审查原始模型实际对图像依赖（Real/Zero/Shuffle/Visual-off）、语言角色与目标/参照文本，及公共地图可用对象边界。不得用匿名目标实例Contour构造推理ROI。

## 下一阶段执行顺序
1. 原始代码/缓存/模型及控制器协议审计；补固定yaw显式实验适配，不改变旧默认；固定源SHA、数据物理实例分组。
2. 基于失败证据确定最小方案（候选方向：地标几何→未知中间区域→目标视觉匹配；是否非残差由测量决定），先TRAIN机制对照，比较geometry/text-only、Real、Zero/Shuffle，证明视觉贡献。
3. 明确方案后真实GPU Smoke/吞吐、seed0小规模训练及独立TRAIN确认。不重复已失败的全局倍率/单质量规则，不盲增大Attention。
4. 冻结模型/配置/128Episode清单后，运行真实闭环Original与新方案配对Val-Seen128。严格>65%且有可信收益才全量；不在同一128反复调参后宣称泛化。
5. 正式Test只在最终冻结后，遵守freeze manifest，不依据Test反馈改模型。没有独立收益则保留负结果、停止扩大。

约束：原始指令保留；GT仅TRAIN损失/事后评分，不进入正常策略；不使用未来图像；既有暂停任务不自动恢复。固定yaw0表示朝东非北朝上，XYZ不变；原生栅格信息上限不因放大而增加。语言left/right/front/behind参考系未知时不硬编世界方向。

## 2026-11-03第二阶段权威更新（覆盖上文“尚未采集”旧状态）

- 固定yaw适配器graphnav/fixed_yaw_observation.py已完成，原正常入口未改。
- 24物理实例/16图/87参考轨迹帧已真实采集；数据server目录：GraphNav/runs/landmark_chain_visual/fixed_yaw_pilot_24_s0。相机yaw0、agent姿态不变。目标标签独立private文件，仅监督/事后评分。
- 同RGB冻结SigLIP2原ROI32/64/128及小ROI8/16/32编码都完成，目录fixed_yaw_features_24_s0与fixed_yaw_features_smallroi_24_s0；大文件不提交。
- 指令角色由当前会话AI整理，非Qwen。任务显示类别元数据，非严格盲解析金标准；保留原文/不确定方向。
- 原ROI22帧无任何ROI，小ROI13帧；GT点几何/深度代理支持但无ROI10→4。近20m候选支持25→38。
- 87帧空间Top1全文3→11，小ROI目标文本13、分尺度11；小ROI全文shuffle11、zero10。视觉性能增益未证明；新分尺度规则不作为已有效方法。共同25帧NE全文77.58→74.42，目标文本90.30、分尺度88.78。
- reports/landmark_chain_visual/pilot24/REPORT_zh.md、所有逐帧JSON、SHAmanifest及24实例图册cases/index.html已生成。相关17 tests passed，源码compile通过。
- 下一步：观测内细粒度位置候选/直接分布头+独立TRAIN训练与确认；不要继续搜索本24例规则权重。尚无新任务训练权重、Goal Belief指标或SR，不得声称>65%。
