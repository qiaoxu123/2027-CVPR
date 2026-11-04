# Changelog

## [polygon-direction-region-diagnostic] - 2026-11-04
### Features
- 全量重扫22,002条TRAIN，保存427条文本/地标候选及逐条作用域决定；265条有效直接关系按地标、Episode和官方实例身份分组。
- 新增完整Polygon局部前缘的射线/45度锥形方向区域基线，支持多片段、连续距离和多候选；冻结原32参数模块对照，未接入普通导航。
### Design Rationale
- 将公开几何推理和私有目标评分进程隔离，先封存选择、组划分与预测；新开发/确认组排除此前154条方向实验曝光组。
- 预先规定开发收益门槛，未通过则不启动新训练；保留负结果，避免按确认集改变坐标、Anchor或规则。
### Notes & Caveats
- 新确认30条来自13个组，地图仍可能重叠；east/west各仅1条且不在独立确认组，不能宣称四方向泛化。
- PolygonCone未超过原32参数模块，故未训练新模型；道路Polygon内部含目标的现象不支持把该区域一律作为排除区。
- 方向地理含义仍未证实；未改PGVB/导航权重、控制器、正式协议，无闭环或Val/Test调用。

## [polygon-rgb-projection-audit] - 2026-11-04
### Features
- 新增48例TRAIN具名Building/Road Polygon原生384投影审查，覆盖8种Yaw、15–100m高度、部分/完整/超视野/画外情况。
- 保存原RGB、源片段及完整同名Polygon并集、几何/有效源Mask、数值变换与ROI/patch阈值诊断及HTML核验记录。
### Design Rationale
- 分开检查连续Footprint的可逆性与实际整数栅格warp，并区分整物体入镜比例和视野占比。
- 不把单个道路片段的画外状态当作整个具名道路不可见；无人工Mask只给几何覆盖和定性意见。
### Notes & Caveats
- 12个超视野样本有足够有效源支持，却被整物体50%/90%阈值全部拒绝；压力样本计数不代表全TRAIN比例。
- 像素中心约定下取整偏差最大0.851px；这不是人工视觉配准精度。建筑边界语义仍需核验。
- 未加载/训练模型、未运行关系推理或导航，未改生产投影、渲染、Yaw、阈值及原Checkpoint；18项相关测试通过。

## [pgvb-structured-landmark-pilot] - 2026-11-04
### Features
- 全量22,002条官方TRAIN指令导出无目标GT参与的结构化JSON，保留实体/span、方向/参考系、完整最短关系链及歧义；增加60条人工核验工具与24条AI诊断记录。
- 新增冻结SigLIP2的Polygon Cross-Attention视觉Belief训练、同图同类困难负例、城市隔离、真实GPU Smoke及轮末恢复。
- 完成12轮5,112步及冻结开发评估，保存覆盖/RGB置零/打乱/Polygon关闭与互换/名称互换消融。
### Design Rationale
- 绝对Polygon中心不进入视觉评分，采用相机尺度去中心轮廓；世界投影独立于几何Belief，避免复制地图位置。
- Cambridge拟合、Birmingham开发，消除相邻地图块同物理地标跨分区；原模型默认入口和Checkpoint不变。
### Notes & Caveats
- 本版同图区分训练94.06%、开发50.28%；开发NE弱于覆盖基线，Polygon形状无稳定收益，不进入关系网络阶段。
- 该负结果仅适用于去中心形状条件实现，未验证完整绝对Polygon配准。Polygon监督是弱代理，非真实可见性或人工像素GT。
- 没有人类语言Gold，不能报告人工解析准确率；所有自动关系均未批准用于训练。未访问正式Val/Test或运行新闭环。

## [geometry-guided-visual-belief-pilot] - 2026-11-04
### Features
- 补充完全不接收几何logits的Visual-only世界Belief接口、覆盖范围/置零/错配对照及数值并列峰测试，单独验收视觉定位。
- 冻结原A3/SigLIP2，新增139649参数的地标视觉Belief头、公开区域弱监督、多视图对比和几何蒸馏训练。
- 补充TRAIN公开地标观察，完成v1及世界蒸馏修正v2、反事实和相同条件64地标闭环/置零/原RGB回归。
### Design Rationale
- 将可训练视觉评分与绝对地图坐标分离，显式区分观测域V与全局G+V；未知观察不构造不存在标签。
- 控制接口保持原raw-logit口径及零证据一致性；蒸馏目标必须与真实世界网格聚合一致。
### Notes & Caveats
- 地标任务有2例RGB相关纠正，尚无显著性或原CityNav目标任务SR结论；Dev一致性/误改未过门槛，未进入第二阶段。
- 同融合的未训练原生评分同为63/64，训练头没有额外成功数；保留这一负对照，不将融合收益冒充训练收益。
- 标签为Polygon几何代理，不是人类像素GT或真实可见性；原权重、正常入口及控制器未改。

## [named-landmark-navigation-diagnostic] - 2026-11-04
### Features
- 新增独立公开地标名称任务E1/E2冻结闭环诊断、Polygon距离评分、可恢复记录及真实RGB轨迹图册。
- 追加严格不执行A3、均匀先验、零几何上下文的仅视觉组及匹配Oracle接口控制；正常推理入口不变。
### Design Rationale
- 将公开地图Goal已知的控制能力与名称到位置预测分开；道路、多碎片地标不使用质心评价，不读取旧CityNav目标点推理。
### Notes & Caveats
- 64组名称级任务不是官方实例目标任务SR；原Checkpoint/默认推理入口/控制阈值不变。E2允许公开具名Polygon，不能冒充纯视觉成绩。

## [landmark-associated-train-v1] - 2026-11-04
### Features
- 新增可复现TRAIN关联筛选生成器、原始/processed兼容数据及后续拟合配置；移除未找到关联和不确定类并保存逐条日志。
### Design Rationale
- 固定全量自动分类，保持原始字段、既有地图分组及完整来源索引，防止版本索引错配和内部验证混入拟合。
### Notes & Caveats
- 筛选不是人工质量认证；原分组曝光历史不重置，正式评测不筛选。大型数据仅保存服务器，本次未训练。

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

## [aligned-height-training-and-instance-label-audit] - 2026-11-04
### Features
- 独立重采样371个TRAIN非初始位姿，保持XY/yaw，恢复1271个逻辑状态并匹配随机批次顺序。
- 新增仅供训练标注侧使用的实例core构造器，其他车辆不因20m邻近就当目标，歧义与无效点保留unknown。
### Design Rationale
- 通过同预算真实闭环区分数据高度匹配与视觉贡献；避免位置成功半径代替实例身份。
### Notes & Caveats
- 重采样版真实RGB/恒灰/打乱同为21/25，尚无视觉净收益。实例标签仅为弱几何提案，尚未用于训练，不是像素GT。

## [weak-instance-ranking-loss] - 2026-11-04
### Features
- 新增仅TRAIN损失可访问的目标core/其他车辆排序项，记录独立视觉梯度与有效监督次数；正常forward不增加私有字段。
- 新增保守Contour背景标签与Car粗位置辅助项替换开关，unknown和无效几何弃权；注册车辆core与背景来源分开统计。
### Design Rationale
- 避免20m位置正区域把相似车辆和背景都作为视觉目标，先检验监督而非增加网络层。
### Notes & Caveats
- 仅core排序使拟合54帧代理命中11→26，但开发真实SR80%低于Original84%；背景版本此阶段尚未训练。全部代理不是人工像素GT。

## [frozen-val-seen128-result] - 2026-11-04
### Features
- 完成首次冻结Val-Seen128配对真实闭环，新增物理目标/地图分组区间、严格全量门槛与8个实际RGB案例。
### Design Rationale
- 不再将重复TRAIN开发小样本当最终导航证明；模型和数据清单在运行前封存，只评估Original及已冻结版本。
### Notes & Caveats
- Original45/128，新模型52/128，+11/-4净7，未达到至少84门槛，未启动全量或Test；点差异不等于显著提升。
- 新实例背景版本仍只在TRAIN测试，未用Val选择它或修改参数；它的开发SR未超过Original。

## [local-relation-factor-pilot] - 2026-11-04
### Features
- 新增独立多参照关系提案、公开地标分支几何与带Null的参照到世界位置因子；原V3参数和普通入口冻结。
- 新增同预算隐藏/显式关系训练、输入白名单和断点恢复Smoke入口。
### Design Rationale
- 检验旧局部视觉输入丢失关系类型是否限制目标定位，区分关系信息与额外容量的贡献。
### Notes & Caveats
- 关系由本地句法模型提出，不是人工实例监督；between、序数和长链暂弃权。实验尚未产出性能结论，不启动Val/Test。
- 后续实测：Hidden/Typed各完成三轮；Typed未超过Hidden，RGB对应打乱未改变最终Goal，作为负结果保留，不接入默认导航。报告保存真实训练和恢复日志、逐状态指标及6个RGB案例。

## [region-reference-supervision-probe] - 2026-11-04
### Features
- 新增公开具名参照的实际窗口覆盖审计、TRAIN损失侧区域类别代理及视觉输入专用线性探针。
- 新增未知区域/边界/类别冲突弃权与不依赖标签的视觉Shuffle对照。
### Design Rationale
- 先检验普通停车区等中间参照的监督与视觉局部性，避免继续用稀疏且有作用域错误的关系提案扩大网络。
### Notes & Caveats
- 代理不是像素GT或可见性标签；探针不是导航模型，不改变原入口和权重。训练结果尚待实测。
- 后续真实GPU实测：固定三轮探针开发Balanced Accuracy80.85% vs原生64.31%、Shuffle57.18%，仅证明弱区域语义可学，没有新Goal/SR；保存完整开发和既有高度范围的事后分层，不删除困难低空状态。

## [region-auxiliary-goal-pilot] - 2026-11-04
### Features
- 新增区域语义辅助损失传入原V3共享图像投影、同起点同预算Goal-only对照及标准部署导出检查。
### Design Rationale
- 将已验证可学的区域信息用于实际Goal训练，避免继续增加关系图复杂度。
### Notes & Caveats
- 辅助头仅训练存在，推理结构不变；区域代理不等于实例GT。预案固定最后三轮，结果尚待运行。
- 实测A/B均82.61% Goal R1低于冻结85.87%，辅助分类虽学会但不晋级导航；新增20m正例混杂审计和仅TRAIN教师侧重配对提案覆盖检查，禁止GT实例关联进入正常推理。

## [retarget-soft-instance-pilot] - 2026-11-04
### Features
- 新增真实TRAIN观察的查询重配对、新指令A0/BERT/SigLIP文本重缓存、私有软实例排序训练。
### Design Rationale
- 扩大目标核心稀疏的训练监督，先核验像素支持和语言条件一致性，不增加视觉骨干/关系层。
### Notes & Caveats
- 粗Car轮廓和2m核心不能冒充像素实例标签；未知复核弃权、其余只作弱偏好。推理结构/协议不变，训练结果待运行。

## [target-instruction-channel-ablation] - 2026-11-04
### Features
- TRAIN实验训练入口新增显式目标语言通道对照，参考通道及默认role行为保持不变；支持同预算GPU Smoke与恢复。
### Design Rationale
- 用原始全文目标向量检验被删实例限定词的影响，不增加网络或修改原始指令，避免同时改变目标和参考通道的混杂。
### Notes & Caveats
- 预训练文本长度限制仍存在；此为待验证对照，不是导航性能改善。旧权重、缓存、控制器和默认推理入口保留。

## [global-reference-evidence-transport] - 2026-11-04
### Features
- 新增显式实验用全局参照证据传递核及同容量Local/Global/固定图像对照，保留原冻结SigLIP2、A0和V3。
### Design Rationale
- 当前局部更新无法重排覆盖外先验；新核允许当前参照视觉通过相对几何影响全局候选，并以中性背景和采样归一化保持零证据/重复采样不变性。
### Notes & Caveats
- 单帧支持界限不等于轨迹SR上限；未验证参照实例身份。GPU逐值初始化核验使用同次基线，排除旧CUDA scatter重复求和差异。普通导航入口和原权重未更改。

## [scoped-between-supervision] - 2026-11-04
### Features
- 研究用二端点between关系与中间区域作用域编码；完整TRAIN公开语言提案及独立目标采集名单。
### Design Rationale
- 保留成对参照、公开同名多轮廓与未知专名，避免丢弃between或伪造数量词所指实例。
### Notes & Caveats
- 依存提案不是人工语义金标准；新增数据不代表导航性能，默认入口、控制器和原权重保持不变。

## [public-geometry-visual-pilot] - 2026-11-04
### Features
- 公开地图条件化全球场与当前观测视觉头，固定三组GPU训练、Smoke恢复、封存后开发评价及RGB失败可视化。
### Design Rationale
- 将地图收益与新增视觉贡献分开；不依赖从RGB识别公开专名，不将局部参照直接等同目标中心。
### Notes & Caveats
- 拟合提升未在开发复现，所有新头均不接入默认导航；没有新闭环或正式评测成绩。

## [ai-weak-vehicle-annotation] - 2026-11-04
### Features
- 48例TRAIN当前AI弱核验、27个弱外观框与车辆组标注；原生坐标/SHA/schema检查及候选格点覆盖审计。
### Design Rationale
- 分开注册位置、Contour、实际RGB身份和目标组语义，不把宽松导航位置标签当像素实例真值。
### Notes & Caveats
- 无人类金标准，完整语言身份未核验；弱框不直接监督实例或Goal，未训练或产生新导航成绩。

## [native-grid-frozen-diagnostic] - 2026-11-04
### Features
- 独立原生144/576共享特征候选诊断、GT-free排序干预及显式隔离的弱框Oracle，保留分数集合和完整失败案例。
### Design Rationale
- 分开候选覆盖、外观排序和最终Goal；同一真实384px前向避免分辨率、批量与重复编码混杂。
### Notes & Caveats
- AI弱外观代理有改善，但Goal正确集合无提升，原生重排净损失；不接入普通导航、不扩大缓存或运行全量。

## [PGVB-local-anchor] - 2026-11-04
### Features
- 新增版本化局部Polygon Anchor、同名道路片段汇总、正区域独立监督与冻结回归入口。
### Design Rationale
- 分离整地标覆盖与当前有效视觉支持；不把未知区域作为负样本，保持既有网络参数结构。
### Notes & Caveats
- 保留原权重及正常推理入口。评估为TRAIN公开Polygon弱监督代理，无人工像素Gold，无闭环导航成绩；单种子且不是新盲确认。

## [PGVB-closedloop-audit] - 2026-11-04
### Features
- 在独立PGVB地标闭环入口中修复整数/字符串目录ID适配、无证据XYZ waypoint维度；新增逐决策RGB与世界Belief只读快照、按方法断点执行及64条配对报告。
### Design Rationale
- 保持原E2模型与控制器不变，以历史逐点轨迹复现核验，再固定比较PGVB增量和独立视觉Goal。诊断快照不回流推理。
### Notes & Caveats
- 公开具名地标Polygon区域任务，不等于原CityNav目标任务。E5无全局搜索、候选Context为零；SR按末位置评价，不能将超时/停滞成功称为正确Stop。不训练或自动部署。

## [PGVB-dynamic-goal-interface] - 2026-11-04
### Features
- 新增独立动态Goal候选确认与网格量化到达接口的冻结权重2×2实验；保存历史50次跳变及TRAIN分组闭环诊断。
- 新增跨Block同城同名地标保守分组检查，保留首轮隔离局限与未评分任务的额外确认记录。
### Design Rationale
- 候选稳定只重排现有候选并gather原context，不改变Belief logits；错误旧候选至多短暂延迟，不能永久锁定。到达接口仍依赖原学习式Stop，不使用GT或Polygon距离。
### Notes & Caveats
- 原默认策略、权重、Stop概率阈值及Horizon20不变。宽松到达存在错误Goal上的提前误停反例，不能据开发集结果直接部署。所有成绩为TRAIN具名地标诊断任务。

## [direction-belief-transform] - 2026-11-04
### Features
- 新增32参数、方向条件的Anchor→Target Belief位移核；支持四个工程世界方向和显式参考朝向下的四个相对方向。
- 完成合成/真实TRAIN同预算无方向对照、公开输入预测封存、原目标GT事后评分及当前冻结PGVB Anchor小样本检查。
### Design Rationale
- 以可解释位移与不确定性验证词是否控制分布，避免直接扩展完整视觉关系网络；真实参考系不确定时拒绝强制转换。
### Notes & Caveats
- 冻结原PGVB/视觉骨干/导航器，不做外观识别或闭环。真实筛选无east、west仅2个评估配对；地图真实地理北尚无CRS校验。真实目标覆盖不足，未证明PGVB Anchor增益，不自动部署。
