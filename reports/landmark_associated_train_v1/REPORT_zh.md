# CityNav 地标关联筛选TRAIN子集 v1

按用户要求移除“未找到明确关联”和“不确定”指令。只使用已有的全量自动分类，不将101条抽样复核的加权估计用作逐条标签，也未单独覆盖抽样标签。没有训练、没有访问Val/Test，没有改写指令或覆盖原始数据。

## 数量

|全量官方TRAIN分类|处理|数量|
|---|---|---:|
|直接几何关系 direct_geometry|保留|9,356|
|直接名称绑定 direct_identity|保留|1|
|间接关系 indirect_geometry|保留|7,182|
|未找到明确关联 no_confirmed_anchor|移除|325|
|不确定 uncertain|移除|5,138|
|总计|22,002 → 16,539|移除5,463|

保留75.17%，移除24.83%。这是自动解析所能确认的关联子集，不是人工核验的高质量金标准。被移除的指令未必无效；很多不确定样本可能是解析器、错拼或共指问题，移除会改变训练分布。

|目标类别|保留|移除|
|---|---:|---:|
|Car|8,349|3,392|
|Building|6,729|1,558|
|Ground|872|356|
|Parking|589|157|

同时为当前训练工程生成processed版本。原processed TRAIN为21,878条，其中移除5,339条（未关联204、不确定5,135），保留同样16,539条。相较官方版少的124条均属于本次移除类；没有把不同版本的源索引混用。

## 后续训练入口与分组

数据目录：

`/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/datasets/landmark_associated_train_v1_provenance/`

|文件|用途|数量|
|---|---|---:|
|official_train_associated.json|保持官方原始字段的完整筛选版|16,539|
|processed_all.json|保持现有processed字段的完整筛选版|16,539|
|processed_fit.json|后续实际模型拟合|15,202|
|processed_dev.json|原TRAIN内部开发组|670|
|processed_confirm.json|原TRAIN内部确认组|667|

后续采用`configs/datasets/landmark_associated_train_v1.json`登记的路径，以`fit_annotations`作为模型拟合数据。该JSON是数据配置清单，不声称现有所有训练CLI已经自动加载此配置。实际开始下一轮训练时需显式传入该文件路径，并重建或校验相应缓存。没有自动恢复旧训练。

沿用既有18/3/3地图fit/dev/confirm分组；地图间及实例ID组间无交叉。这些dev/confirm保留此前被使用过的历史，不能重新宣称为未使用过的独立确认集。筛选只改变TRAIN，正式评测仍保持完整清单及原协议。

## 记录与恢复

- `removed_records.jsonl`：5,463条官方源索引、原文、地图、实例ID、类别、移除原因及原分组；原始数据未删除，可按索引恢复。
- `retained_records.jsonl`：16,539条保留记录及官方源索引到子集索引映射（大型文件保留在服务器）。
- `official_selection.csv`：22,002条完整保留/移除决策索引。
- `processed_selection.csv`、服务器`processed_index.jsonl`：processed源索引、筛选后索引、所有相同map+instruction的官方索引候选。重复文本不伪造唯一官方索引。
- `manifest.json`：源数据、分类文件、地图分组、生成脚本及各产物SHA256。
- `verification.json`：逐条核验保留行全部原始字段不变、输出SHA一致。

过滤后Episode顺序已变化，不能将旧缓存中的第i条直接视为新数据第i条；应按来源映射或稳定map+instruction/实例ID对齐。GT字段和轨迹仅原样保留在TRAIN标注中，不因此加入正常模型推理输入。RGB、yaw规则、权重与模型入口均未修改。

## 验证

生成器完成全部官方及processed任务检查；逐条比较两版各16,539条保留记录的全部字段，均与对应源记录相同。全部生成文件SHA已核验。相关3项测试通过，覆盖筛选策略、未知分类拒绝、跨版本重排序连接、字段保留、原分组保持及禁止覆盖已有数据目录。

数据本体约396MB仅保存在服务器；Git提交生成器、配置、统计、移除日志和完整决策索引。首次构建的原型目录`landmark_associated_train_v1`保留但不作为训练入口；正式路径以配置中带`_provenance`的目录为准，补全重复文本的多来源索引。
