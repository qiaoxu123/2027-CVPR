# Polygon 关系学习：冻结实验设计

基于 PR16 `1ac37c9`。不修改任何正常推理入口或历史权重；不访问 Val/Test。

1. inside/outside 用完整 Polygon 的带符号边界距离生成软先验；near 定义为距离 Polygon 不超过30m（包括内部），尺度预先固定，非官方语言距离标签。
2. 未获得独立 CRS/地理控制点前，不训练或评价地理方向正确率；north/south/east/west/along 回退 Geometry，不用目标GT配准。
3. 复用493 fit /80 dev 的 TRAIN 地图及物理地标分组；dev此前被诊断过，不能称全新盲测集。今回固定6轮、不用 dev 选 epoch/混合权重。
4. 从单一公开对象绑定的 TRAIN 指令构造inside/outside/near三种文本；BERT全文重新编码，结构Predicate同步替换。空关系区域或空补集不强行造标签。类别损失逆频率平衡，不重复伪造样本。
5. 仅训练原DirectTargetHead的word/predicate/relative/relation_score。合成反事实使用均匀正确区域CE与margin=0.2概率排序损失；原始inside/near指令使用15m高斯目标坐标CE；另有0.2 Geometry KL保真项。反事实没有新目标坐标。
6. 四方法：Geometry原始；Explicit纯几何先验；Relation=Geometry+训练残差；Fusion=Relation与Explicit归一化概率各50%。不支持的关系四组都回退原Geometry。
7. 输出自然指令目标R20、NE、mass20、相对Geometry纠正/破坏；反事实inside/outside共同组响应、near区域响应与质心位移。inside/outside没有固定平移方向，不报告伪造方向正确率。
8. 本轮不自动启动56条Val闭环；先判断TRAIN净收益及破坏。
