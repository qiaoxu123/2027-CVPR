# 语言关系—Polygon方向预测研究计划

范围：仅Revised TRAIN、冻结BERT，不使用RGB、GeomBelief或导航器。原有8方向头14%左右留出成绩为历史背景，不能直接作新划分上的公平对照。

## 数据与关系

复用PublicNames盲绑定，spaCy依存图提取target/中间实体/具名地标、带主客体方向的关系链。旧classify函数曾由GT类别/ID辅助选主语；新封装仅传category=None、identity_ids=()，无法确定则unknown，不使用GT修正语义。保存原文片段、跨度、路径方向、歧义及自动分类。开发组人工文本抽查，禁止以位置真值反推关系。

公开几何为完整Polygon，保留全部语言引用Anchor，额外上下文按公开Polygon间边界距离与固定类别配额选取，防止近处车辆占满。邻居名称、轮廓采样、相对XY、方向、距离、包含、交叠和邻接均来自公开地图，不用目标GT选对象。道路区域不当作中心线。

## 独立划分

原5张已分析地图作为development；从其余地图按city分层、SHA("relation-confirm-42"+地图名)排序，每城取2张作为新confirmation，余下fit。划分先于任何新分数。地图、目标ID/公开轮廓、重复规范化指令隔离。所有新模型从seed42随机初始化；不能将旧模型（曾拟合新confirm地图）直接用于确认组。

## 受控模型

A：原NeighborhoodBearingHead重新拟合相同fit集合。
B：新的命名Polygon对象匹配、拓扑/形状表示，不用解析链。
C：在B上加入anchor→中间实体→目标的语言关系路径条件传播。

已知Anchor身份仅来自公开名称匹配；目标/中间实体保持概率匹配。仅当TRAIN真实目标ID恰好在事先公开选取对象列表中，才使用精确对象ID作目标匹配损失；不使用GT最近距离制造实例标签，也不给中间实体虚假标签。方向标签为GT坐标相对Anchor顶点均值的atan2，不是front/left语言参考系真值。无上下文目标时预测自由角度分布/未知概率，不注入GT证据。

主监督：8方向CE；匹配辅助CE0.5、公开上下文是否含GT对象BCE0.1（B/C同监督）；query等权。无语言方向→世界轴硬规则。输出8方向分布、熵/NLL/ECE不确定性。

## 开发与验收

先合成几何与GT隔离测试、真实GPU Smoke；开发阶段固定seed42、hidden128、AdamW、clip1，最多12epochs，按dev query-macroTop1选epoch（并列按NLL），保留所有训练日志/负结果。

开发组分别评估direct/indirect/unknown、single/multi、打乱语言/解析链与几何消融。若结构化模型无稳定收益，先按开发证据纠错，不提前读confirm分数。方案冻结后一次确认，重训A/B/C公平对照，按目标与地图分组bootstrap、配对纠正/破坏与逐图效果验证统计意义。仅此后可判断阶段目标是否达到；无正证据不得宣布完成。

每阶段Commit/Push。保持原权重和正常推理入口不变。不启动导航，不访问正式Val/Test。

## 开发期公平性补充（确认组评分前固定）

为了不将额外TRAIN实例ID监督收益误称为网络/解析收益，再完整训练B/C的angle-only版本，模型和特征/初始化/预算不变，仅去掉对象ID匹配及覆盖BCE。主比较采用旧8方向头 vs C-angle-only（同样8方向标签），B-angle-only vs C-angle-only隔离解析链贡献。原B/C-instance保留为辅助监督消融，不能用更高的一组挑选后冒充相同监督比较。确认前只依据开发数据选各自epoch，所有版本均报告，确认不用于改结构或调参。
