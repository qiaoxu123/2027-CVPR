# PR18完整地标RGB：TRAIN特权输入实验

- 基准ef19a7f；独立experiment/complete-landmark-rgb-training-20261105。修复probe的真实数据属性与Top2接口，但不运行Val探针。
- 全部21878条Revised TRAIN进入清单；同一公开引用集可复用图像。无引用/非法Polygon/无有效源像素保留静态回退并记原因。输入区由全部合法公开引用Polygon及固定25%上下文确定，不按目标GT或得分选图。
- 地图/城市隔离：Cambridge14259条fit，Birmingham7619条Dev；这轮PGVB不得拟合Dev，冻结核心曾训练全TRAIN，因此不宣称全模型未知地图泛化。
- Whole：384全景。Multi：同一全景范围内64m/128m上限分块，筛选与Polygon.buffer(32m)相交的局部块，保持小目标源像素。多尺度来自同一正射纹理，不是新视角；按世界格平均证据，不能重复累加置信度。
- 对照Static、当前真实起点RGB（统一公开引用接口）、Whole、Multi；旧唯一Anchor接口单独统计。视觉门控均关闭（离线图像内容诊断），不是tracked控制器SR。
- SigLIP2/BERT/Geometry/Relation/控制器冻结；缓存真实编码，先冻结比较，再训练PGVB；另训练已有joint_logits作为视觉融合Head，joint_context不训练、保持零映射。固定3轮、最后Checkpoint，不基于Dev选轮次/阈值。
- 训练只用fit目标坐标；Dev目标仅在预测封存后评分。未知实际可见性保留未知。R20/NE/TopK是位置指标，不是像素实例识别率。
- 特权checkpoint禁止进入正常导航。若明确优于静态、当前RGB且纠正>破坏，才另建因果观测版本；本轮不自动拿特权模型做Val闭环。
- 依PR18文档，原始特权底图裁剪留服务器，不推送许可未确认的源图。仓库交付采样几何图、Belief图、指标、HTML源及复现脚本；服务器HTML含真实RGB例子。

- 固定训练损失：目标15m软标签CE，仅在当前采样有效世界网格20m内有支持时启用；其余仅保真KL，KL权重0.2。AdamW PGVB3e-5/joint1e-4，batch4，裁剪1.0，seed42，固定3轮。支持判断仅作TRAIN损失权重，不参与裁剪/推理。

- 全量公开同名/同类型片段补齐：旧容量截断的71组引用、219条fit查询更新，Dev输入保持逐项一致；新清单inventory_all_public_parts，59800视图条目（不冒称独立纹理/视角）。
