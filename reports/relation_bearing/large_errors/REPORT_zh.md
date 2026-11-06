# 大于90°方位误差：指令与Polygon事后审查

2026-11-06。仅分析上一轮已封存TRAIN确认预测；没有改模型、权重、Polygon绑定或坐标规则，没有新训练，也没有Val/Test/RGB/导航实验。图上GT仅用于事后定位和评分。本报告不能再被用来改方案后宣称同一确认集是新独立验证。

## 数量与分布

10,451个指令—Anchor对中**2,370对（22.68%）**误差>90°，涉及1,595/3,599条可评分指令、542/712个登记物理目标。一个指令可有多个Anchor和重复描述，不能将2,370当作独立任务数，也不能将“某目标曾有一个错误”视作该目标所有指令都失败。

| 分组 | 大误差 / 总预测对 | 大误差率 |
|---|---:|---:|
| Building Anchor | 862 / 4,168 | 20.68% |
| TrafficRoad Anchor | 923 / 3,273 | 28.20% |
| Footpath Anchor | 542 / 2,779 | 19.50% |
| 直接关系（自动代理） | 891 / 4,280 | 20.82% |
| 间接关系（自动代理） | 286 / 1,218 | 23.48% |
| 关系未知 | 1,193 / 4,953 | 24.09% |
| 登记目标对象在公开上下文中 | 1,597 / 7,636 | 20.91% |
| 登记目标对象不在公开上下文中 | 773 / 2,815 | 27.46% |

Birmingham12为442/822（53.77%），Cambridge9为965/4244（22.74%），Cambridge13为960/5371（17.87%）；Birmingham11仅14对，3对错误。地图分布差异明显，不能把错误只归因于复杂语言。

## Polygon本身有无明显绑定/中心问题？

- 全部绑定轮廓与公开原始Polygon经过world_bounds归一化后，最大坐标误差仅1.31e-6；Anchor Polygon有效性检查无无效对象。这支持读取/归一化一致，**不能证明语言指向的就是正确地图实例，也不等于像素配准准确**。
- 有同名多个公开Polygon：1,211/6,041（20.05%）；无重名：1,159/4,410（26.28%）。本轮数据不支持“同名多Polygon是主要大误差来源”。这些是不同样本构成的相关统计，不是控制实验。
- 跨度>100m的Anchor：249/997（24.97%）；较小Anchor：2,121/9,454（22.43%）。长道路有一定风险，但并不能解释全部错误。
- 将Anchor顶点均值换成面积质心，仅16/10,451对的GT方位会改变>45°，其中7对属于本批大误差。**普通质心计算差异不是主要根因**。这里没有实际改变推理Anchor。
- 119个大误差对的GT处于Anchor区域内部；整个确认集有455个这种对。此时“从一个中心指向区域内部点”的角度可能不适合作为区域关系真值，但没有因此删样或改分。
- 上述只检查顶点均值与面积质心，**没有排除长道路全局中心与局部语言参照点不一致**的问题，后者仍需语义证据。

## 典型原始指令与实际地图

[HTML图册](index.html)展示12个实际案例，黄色为公开Anchor Polygon，绿星为TRAIN GT，绿色箭头为GT方位，红色为冻结8方向预测；紫色仅表示潜在对象分布Top-1，不是RGB识别框。所有坐标为工程World XY，不假设北/东映射。

1. **12810，Regent Street（TrafficRoad，跨度约71×122m），误差179.99°。**
   原文：`South of Regent Street near Downing College, this blue roofed building is first one from the left.`
   当前绑定到道路片段580。真实目标登记对象没有进入公开64对象上下文，潜在匹配选中Starbucks。指令同时依赖蓝屋顶、序数、left参考系；模型没有RGB外观，也未核实left参考系。不能仅凭文本south就按World XY翻转标签。这里有上下文缺失和目标消歧不足的直接证据，Polygon并非读错或归一化错误。

2. **10056，Round Church Street（Footpath，跨度约86×75m），误差179.72°。**
   原文：`The white and gray building on Round Church Street across from Cambridge Union Society.`
   GT对象在公开上下文，模型潜在Top-1却为**Cambridge Union Society本身**，而不是across from的目标。该例符合目标/参考对象混淆；并非正确目标从未提供。不能将这个单例推成全部错误的主因：整个大误差组潜在Top-1为指令任一具名参考地标只有78对，选中当前Anchor只有14对。

3. **13714，Butterfield Café（Building，约24×21m），误差179.74°。**
   原文：`A car parked behind Butterfield Cafe near the Maitland Robinson Library. It is a red car with a tree to one side and two white cars on the other.`
   目标在上下文，但潜在匹配选错普通对象；behind的物体参考系未明确，树和车辆颜色/排列需要当前实验没有的视觉信息。紧凑建筑Polygon仍可出现近180°错误，说明问题不限于长道路。

4. **13299，Tennis Court Road（Footpath，约36×61m），误差179.79°。**
   原文：`A black car next to a gray van, parked in between a multi-colored beige and white building, with a beige roof that has skylights on it and the central delivery building, on the edge of the map off Tennis Court Road, across from the Howard Lodge building`
   GT对象在上下文，预测几乎反向。解析器对该Anchor形成`car → edge of → map → off → Tennis Court Road`，这不等于已可靠建立车辆与局部道路的几何约束；主要between建筑和Howard Lodge关系不能被这条链充分表达。是作用域/中间实体建模不足的具体例子，不是直接east/west符号错误。

5. **10031，Trinity College Chapel（Building，约69×17m），误差179.75°。**
   原文：`The triangular building next to the Trinity College chapel has a rectangular lawn behind it.`
   目标距Anchor顶点均值仅16.5m，潜在匹配选中Great Court E,F,G,H,I and K。`next to`不能唯一给出径向方位，三角形、草坪及behind的作用域对消歧很重要。此例提示应检查目标身份与形状使用，而非直接增加方向分类损失。

6. **10272，Jesus Lane（Footpath或TrafficRoad两个片段），误差约179.9°。**
   原文：`A large building with an inner court yard and a slightly taller square tower section on Sidney Street at the end of Jesus Lane.`
   当前链把`a slightly taller square tower section`作为中间实体，并含不可靠with关系，已标不确定。模型潜在选择Squash Court。道路端点、庭院/塔楼结构并没有被通用方位平均充分表达。重复片段分别保留，不用GT选最近片段。

## 冻结分支重放：错误在哪一层出现？

重新加载原封存Checkpoint，只输入原公开特征，逐项确认重放概率与已保存预测最大差<1e-5；GT仅对分支输出评分。

- 2,370个错误对的几何分支混合系数均值**0.99146**。这是未校准的混合系数，**不是正确目标存在概率或99%识别置信度**。
- **2,356/2,370（99.41%）**在对象概率→公开方位映射分支单独输出时已经误差>90°。
- 自由角度分支单独有1,441对>90°；两分支同时>90°为1,427对。分支输出只是归因，不作为新方法结果，也没有选择融合系数。
- 1,597个上下文含登记目标的错误对中，只有155对潜在对象Top-1为该登记目标。其余1,442对潜在匹配选了别的对象。这支持**目标对象概率识别/消歧不足**是重要问题，但公开目标ID监督自身有不确定性，不能称为人工核验实例错误率。
- 155个“对象Top-1为登记目标但方位仍大错”的对中，21对GT在Anchor内部、16对目标GT距登记对象轮廓>1m；还可能涉及目标面积内位置、对象分布多峰/边界方位核、最终混合。Top-1对象相同不意味着整个对象分布或角度核正确，剩余原因不能仅凭这些标量确定。
- 主模型仅有方位分类监督，潜在对象身份没有受到精确ID监督，存在不可辨识性：其他同方向对象也可能满足训练目标。因此上述对象匹配统计用于分析概率分布，不能直接证明一个独立实例识别器的准确率。

## 当前判断与下一步

**现有证据不支持用统一坐标翻转修复，也不支持把主要原因归于Polygon顶点均值计算。** 证据较强的是：公开对象空间先验被高度依赖，但目标身份概率经常错误；蓝屋顶、白灰建筑、车辆排列、道路端点、behind参考系等约束尚未被可靠映射到对象。另有773个大误差对连登记目标对象都不在当前公开上下文内。

下一步先在开发数据审查/验证**目标—参考角色与候选对象覆盖**，区分“只有near无法唯一决定方向”和“明确可执行关系被解析/匹配错”。对长道路保留局部片段与不确定方位；不要根据GT改绑或翻转World轴。暂不调整融合系数、不增加网络、不重训、不将这些确认集案例用于调参后再次宣称独立增益。

保留所有2,370个大误差对、完整指令、Polygon、解析链、公开上下文状态和分支重放输出，见本目录JSONL。原始预测和Checkpoint未覆盖。
