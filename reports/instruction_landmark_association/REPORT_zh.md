# CityNav TRAIN：目标与公开地标关联方式分析

**结论：绝大多数指令可以利用公开地标，但“直接关联”与“先定位停车场/车排/参照物再定位目标”都占很大比例。当前会话101条分层语义复核加权估计，约51%直接关联、46%间接关联；约1%缺少明确可绑定的地图锚点，约2%仍不确定。这些是抽样校正估计，不是22,002条逐条人工金标准。**

## 数据与定义

完整官方TRAIN 22,002条任务指令，21,303个唯一map+instruction，24张地图；未使用Val/Test构建统计。逐任务原文、图关系与匹配名保存在records.json/CSV。仅使用原文、公开CityRefer名称/类型及TRAIN目标类别/实例标注辅助确定分析对象；**没有用目标GT坐标或轮廓筛选地标**，没有模型训练或导航。

“地标”在本报告指可绑定到该地图CityRefer目录的具名道路、建筑等公开对象。未具名的树、汽车、房屋、停车场是普通参照物；它们可以充当中间节点，但不自动等于可定位的公开地图锚点。

- 直接：目标与具名地标有明确空间边，如building on St Peter’s Street；几何部位词side/front/corner不当成独立物体。直接叫出目标自身目录名另列identity。
- 间接：目标→中间对象/区域→具名地标，例如车→停车场→One Stop，车→车排→停车场→Nike。路口或未知车道等中间构造也计入当前多跳定义；并非所有间接链都必须依赖视觉。
- 无明确锚点：未发现可靠的公开名称绑定，且无明显待核对专名线索；常见只有颜色、车排、邻车、树木。它不等于绝对无法与地图关联，RGB/区域识别仍可能帮助绑定。
- 不确定：名称缩写/错拼待核对，或关系作用域/共指未解析；不能将算法解析失败当成指令无关联。

## 全量自动初筛（保守解析结果，非全量语义金标准）

|类型|条数|占全部TRAIN|
|---|---:|---:|
|直接几何关系|9,356|42.52%|
|直接自名称绑定|1|0.005%|
|间接关系链|7,182|32.64%|
|未发现明确地图锚点|325|1.48%|
|名称或关系作用域待确认|5,138|23.35%|

20,959/22,002（95.26%）包含可字面绑定的公开名称。自动图中1,598条同时存在直接和间接路径，按最短可靠路径归到直接；两种特征不能直接相加。

自动未判明比例较高，主要是复杂句、共指、拼写变体以及4跳路径截断；抽样中多数仍有明确地标关系。所以不能把23%左右的自动uncertain写成“23%指令根本无法关联地标”。

## 当前会话语义复核及校正估计

解析器冻结后，以新seed分层抽取101条，我直接阅读原始英文、解析链与该图公开名称进行复核；没有调用Qwen或外部LLM API。复核不是人类标注，也不能保证零错误。20条自动uncertain中14条可归到间接、5条直接，1条仍保留不确定。20条无锚点样本中，有3条其实能通过名称变体建立关联，另1条疑似错拼仍未知。

|复核校正后的类别|估计数量|估计比例|分层bootstrap95%区间|
|---|---:|---:|---|
|直接关联（含自身名称）|约11,175|50.8%|43.0–58.9%|
|间接关联|约10,055|45.7%|37.4–53.8%|
|缺少明确可靠地图锚点|约260|1.2%|0.9–1.4%|
|仍需确认|约513|2.3%|0.0–5.8%|

区间只反映该分层样本的抽样波动，不包括本会话复核错误、解析设计偏差或其他地图分布偏差。建议将结论表述为“约一半直接、接近一半间接”，不把点估计当成精确全量人工统计。

## 目标类别差异（全量自动初筛）

|目标类别|总数|直接含名称|间接|无锚点|待确认|
|---|---:|---:|---:|---:|---:|
|Building|8,287|5,185|1,544|44|1,514|
|Car|11,741|3,144|5,205|261|3,131|
|Ground|1,228|580|292|7|349|
|Parking|746|448|141|13|144|

车辆有5,205条已经解析出的间接关系，明显多于其3,144条直接关系；建筑则直接关系为主。不能直接把这些自动类别比例当作抽样校正后的类别内真实比例，因为抽样未按目标类别联合分层。

## 真实例子

- Episode 7797：`The building is on St Peter's Street near the intersection of Pound Hill and is next to the corner building with a white car parked across the street from it.`
  直接：building → on → St Peter’s Street。
- Episode 1027：`The black car on the end of the fourth row of cars in the One Stop parking lot. It's next to the split white arrow on the road.`
  间接：黑车 → 第四排/停车场 → One Stop；不能把Goal直接设为One Stop。
- Episode 1911：`The grey car furthest to the right of the bottom row of the ninth parking section of the large parking lot.`
  无明确具名地图锚点：只能先靠停车区布局和序数，再用观测识别区域。
- Episode 6236：`Grey car in the left parking lot of the west midlands police custody suites in the second lane. It has a blue car to the left.`
  名称变体：police custody suites与目录West Midlands Police Custody Suite可关联，字面失败不代表无关联。

## 对原始GeomBelief的意义

1. “没有任何地标信息”不是这批TRAIN指令的主要共性；优先利用已有地标，避免不断扩展模型却忽略关系作用域。
2. 直接关联不等于目标已经精确定位：on某条道路可能仍对应上百米范围；left/right/front/behind参考系不明确时，不能强制转换为世界方向。
3. 间接任务尤其车辆需要保留“地标→区域→车排/邻近实例→目标外观”的层级。最值得先改的是语言角色/关系作用域和中间区域表示，而不是把停车场或地标中心直接当车辆目标。
4. 此次只说明语言结构分布，不证明该模块一定带来SR收益，更不能替代导航失败归因。无名区域也不等于完全无法关联：可观察RGB可能将其绑定到地图位置。

## 复现与边界

独立分析分支research/instruction-landmark-association-20261103，旧GraphNav主工作树及其未提交恢复修改未动。CPU spaCy依存解析（版本见summary.json）仅作结构提案，本会话复核仅101条；没有训练任何导航模型。9项基础解析、归一化和身份约束测试通过。所有名称关联均来自同地图公开目录，不使用GT邻近检索；没有检查实际地标距离或图像可见性，因而“附近”仅指指令语义参照，并非已验证50m/100m邻域。

运行scripts.analyze_instruction_landmark_association生成全量图，scripts.report_instruction_landmark_association保存复核、估计、报告和HTML。模型环境独立保存在Experiments/instruction_audit_env，不修改原工程Python环境。