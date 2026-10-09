# 当前 Static Geometry 与 SBFNav 输入和处理审计

审计日期：2026-10-09。当前静态实现：`7546f38`；本地 SBFNav 独立复现：`1676759092c2c9aefb419a46f729c708faf8ed2a`。论文：https://arxiv.org/html/2609.05841v1 。本次仅代码/数据审计和规则合成诊断，没有追加模型训练、RGB 实验、导航或 Test Unseen 读取。

## 判断

当前 S2 比 SBFNav 简化了整个信息与处理链。RGB 是一项重要缺失，但名称引用元数据、完整地图上下文、token 级语言交互、候选相对几何、候选多样性与独立重排序都存在差异；固定几何 Prior 与规则还存在可复现的锚点偏置。现有结果不能量化 RGB 的独立贡献。

## 输入和结构逐项对照

| 环节 | SBFNav 论文/本地复现 | 当前 S0–S2 | 影响 |
|---|---|---|---|
| 观测 | 累积已观测 RGB、由 depth 得到的高度；未观测区域屏蔽 | 完全无视觉/高度 | 颜色、车列、材质、屋顶与地面类型缺少证据 |
| 静态地图 | 224×224 全部可用命名地标轮廓层 + 被引用地标填充 mask；本地 catalog 包含 Building、TrafficRoad、Footpath 等真实面 | 仅被引用且严格匹配的 Building；无完整地图轮廓层 | 丢失真实道路面与周边结构，难以定位“在某路上/路口/停车场旁” |
| 历史与位姿 | 当前 footprint、explored、历史轨迹、agent xy；每步更新 | 静态单 Query，无位姿和历史 | 无可见性上下文与新证据更新；本次实验按要求关闭 |
| 语言 | 冻结 SigLIP 的完整 token 序列；地图特征 query token 的 gated cross-attention | 冻结 BERT CLS 单向量加到所有 grid query；稀疏正则解析关系 | target、anchor、属性、序数的细粒度交互被压缩；未匹配名称无法形成显式关系节点 |
| 逐候选几何 | 每个候选与每个地标的 dx/dy/d/sinβ/cosβ，配对名称 token；另有 agent→candidate 几何 | GAT 主要为地标→地标/关系→锚点；grid→landmark 没有直接相对几何特征 | 学习依赖绝对坐标与稀疏规则场，跨地图泛化受限 |
| 形状与场预测 | 残差 CNN 保留 raster 邻域/形状，token cross-attention 后卷积场解码 | 建筑节点为中心、bbox、周长、面积等10维摘要；逐格 MLP residual | 缺少 dense 地图空间上下文与邻域形状特征 |
| 候选 | NMS 多峰，主配置 K=16，追加真实地标质心，去重；本地复现还加 STAY | 普通 topk 全域 grid centers；无 NMS/质心补充 | 前8点可能挤在一个峰；与 SBF Top16+centroids 不同候选预算 |
| 精选 | 候选 RGB crop、指令 tokens、逐候选地标几何进入 Transformer selector；selector 不接收 field score | 直接按 prior+rule+residual 同一组 logits 排名 | 缺少利用独立细粒度证据纠正粗场排序的阶段 |
| 损失 | metric Gaussian field CE + selector listwise/margin/nearest-hard；还有高度/进度辅助 | 仅 normalized grid Gaussian CE，sigma=.9 cell | 不同20m指标对齐方式；没有候选排序监督 |
| 训练 | 本地复现20 Epoch，每个 episode 4个观测前缀 | 5 Epoch，每 Query 1个静态记录 | 当前是预实验；训练暴露、输入和结构同时不同 |

## 名称和引用元数据：先修输入覆盖

以下反事实均只统计同一批当前 Query 的输入覆盖，不使用目标距离，不是训练或定位结果。

| 方法 | Seen：至少一个建筑地标 | Unseen：至少一个建筑地标 |
|---|---:|---:|
| 当前原版引用+严格绑定 | 28.97% (711/2454) | 27.85% (782/2808) |
| 原版引用+限定别名规范化 | 46.94% (1152/2454) | 52.99% (1488/2808) |
| Revised 引用元数据+严格绑定 | 55.83% (1370/2454) | 59.86% (1681/2808) |

限定别名仅正规化标点、前置 the、尾部 building/buildings；仍只匹配源 Building，保留所有同名候选，不做 edit-distance 强制猜测。实际能够补上 One Stop Building→One Stop、BMTR building→BMTR 等。覆盖率改善不能等价于绑定正确率，需抽查名称语义。
当前若保留全部真实类型但仍用严格名称，Seen/Unseen 至少一个地标覆盖即为74.74%/71.47%，说明大量丢失信息本来是道路/步道面，不能全部归咎于 RGB。SBF 本地复现对原版名称的 edit-distance fallback 可达到98.17%/97.97% 返回覆盖，但它没有拒绝阈值，而且每个精确同名只选最大面积面；高返回覆盖不代表身份可靠，不能直接照搬。
当前实验使用原版 CityRefer processed_descriptions；SBF 的 Revised 元数据在共同 episode 上改变了 Train 12327、Seen 1276、Unseen 1464 条地标引用列表。当前原版/本地 Revised split 在共同 episode 上的实际指令文本变化为0，区别主要在引用元数据与 episode 过滤。本地资产事实与论文“revised erroneous instructions”的描述应分开陈述。
原版 Train/Seen/Unseen 为22002/2498/2826；SBF 本地 Revised 为21878/2470/2697。当前转换还从 CityRefer objects 中取指令，SBF 从 split descriptions 取指令；原始逐字差异539/68/69条，大部分是空白，空白规范化后仍有29/2/4条差异。后续应以选定 split 的 instruction 为权威并统一去重规范。不要无说明替换当前数据或历史指标。

## 道路：缺的是中心线和可靠拓扑，不是所有道路几何

已有 CityRefer TrafficRoad/Footpath 是真实 world XY 表面多边形，SBF 将其作为地标面用于静态轮廓/引用 mask，无需道路中心线或可导航拓扑。当前 building-only 协议保护了“不得虚构道路”约束，但同时丢弃了这些有效面证据。可在独立新变体中增加 road_area/footpath_area 实体与直接 surface distance/coverage；必须保留实体类型、坐标与来源，不把面边界当中心线、不凭2D交叉制造 roads_connected。
RGB 应来自真实已观测地图区域/视角，并与公共地图对齐，不得用全图未观测像素、目标裁剪、未来帧或 GT 选图。加入道路面不需要先运行 RGB/闭环。

## 可复现的几何评分偏置

`prior=-min(distance_to_polygon)/sigma` 在建筑内部统一为0。`near` 同样使用内部距离0；`east` 只对中心半平面给分，不排除锚点内部。Prior与rule直接相加，会偏向目标锚点本身。
合成诊断A：建筑[100,100]–[140,140]，指令 east of A；建筑内部东半部(130,120) S1=-0.513，东侧边界外20m的(160,120) S1=-0.984，前者排名更高。
合成诊断B：两建筑中心(100,200)/(300,200)，指令 between A and B；A内部、中点、B内部的规则分都为0，Prior使S1为[0,-3.6,0]，两端排在中点前面。这不是数据性能估计，是对当前代码的精确数值复现。between不要求绝对中点，但目标与锚点不同的任务不应自动偏好锚点内部。
应将区域 Prior 与 relation-conditioned 目标场分开，加入目标/锚点区分和外部/内部适用语义，使显式规则能改正 Prior，而不是只叠加对锚点的偏好。是否排除内部取决于target类别和语言关系，不能对所有任务硬排除。

## 额外几何处理差异

当前 normalized XY 对每个地图长宽分别缩放，y=0在南侧；SBF保持米制比例、正方形padding、y=0在北侧，并使用valid mask。birmingham_block_12长宽比5.46、block_10为4.70；不能把当前张量与SBF/正射RGB直接拼接。当前内部坐标并未因此错位，但接入RGB、复用SBF权重或跨系统比较时必须显式投影/padding/翻转。
当前 .9cell Gaussian 在400m地图约12.9m，在73.3m窄轴约2.36m，和SBF本地的物理sigma20m不同。应统一metric目标场后比较。
当前建筑的 `axis` 是bbox宽高构成的对角方向，两分量非负，不是PCA主轴/真实朝向；center是顶点均值，不是面积质心。654个命名建筑中5个源多边形无效；对649个有效面的两种中心差异均值1.33m、最大13.03m。对无效面，盲用shoelace质心可能非常异常，SBF本地实现也有这一风险，应先做几何有效性校验。

## 建议按顺序补齐

1. 锁定与SBF比较的数据版本/共同Query，并审计Revised地标引用；修复限定别名和指令来源规范。
2. 在新变体中保留真实道路/步道面与全部可用命名地标上下文，不引入虚构中心线/拓扑。
3. 修复Prior与between/east/near的语义偏置；增加候选→每个锚点的相对几何、真实shape/axis特征。
4. 保留完整语言tokens，用token级cross-attention条件化每个空间位置/候选，正则关系作为补充而非唯一显式语言通道。
5. 将SBF地图—语言模块做同输入静态控制组：仅公共地标轮廓和引用mask，关闭RGB/高度/位姿历史；同一Query、28×28世界位置、20m指标和5Epoch预算。分别报告raw-grid Recall与NMS候选覆盖，并匹配候选数。
6. 再增加独立静态selector与ranking监督，最后在相同受控协议中加真实观测RGB，以隔离视觉贡献。Trajectory、Stop和完整闭环继续保持关闭。

## 证据定位

- SBF planning-state：`SBFNav-CityNav-Reproduction/sbfnav/planning_state.py:19,224`；九通道与observed遮罩。
- SBF names：`sbfnav/dataset.py:111,154`；最大同名面、normalized+Levenshtein无阈值fallback。
- SBF token融合：`sbfnav/models/belief_field.py:73`；候选相对几何与selector：`sbfnav/models/selector.py:19,221`。
- SBF候选：`sbfnav/candidate_proposal.py:29,113`；本地训练配置：`configs/sbfnav/main.yaml`。
- 当前CLS：`graphnav/foundations.py:41`；当前候选评分：`graphnav/static_localization.py:240`。
- 当前类型过滤/文本来源：`scripts/prepare_citynav.py:72`；当前规则：`graphnav/semantic_geometry.py:209`。
- 当前shape/axis：`graphnav/semantic_graph.py:21`。
- 审计JSON位于`reports/sbf_gap_audit_20261009/`；可复现审计脚本与日志保留于`runs/sbf_gap_audit/`。
