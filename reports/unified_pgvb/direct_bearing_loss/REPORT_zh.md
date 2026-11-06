# PR26 主Belief Angular Consistency Loss：完整训练与闭环

## 实现和监督边界

A仅Predicate，B加Angular0.1，C加Angular0.3；全部21878条Revised TRAIN，seed42、5Epoch、batch8、Geometry LR1e-5/Relation1e-4、KL0.05、原Goal CE，FP32/TF32关闭。三组实际初始化hash相同，原BERT/PGVB/Stop/Controller不变，不使用PR25角度弱头。Angular只在forward之后监督主784网格Belief诱导的8方向分布，不输入GT角度/目标坐标/距离到新模型。所有任务保留Goal/KL。

TRAIN及runtime统一原文+完整公开目录盲绑定。修复句尾规则全TRAIN0标签、别名、否定/复合方向/朝向/明确间接对象；支持可靠前置方位和后续句子。自动语法筛选不是人工真值。同名多Polygon、多Anchor及模糊关系不参与Angular，但不从数据/导航分母删除。修复无有效候选方位的假常数监督、Angular断点统计丢失；Experimental Checkpoint必须匹配runtime标志。Predicate映射0默认/1east/2west/3north/4south，原8条Embedding不变，缺省与全零输出位级一致。

数据审计：{'counts': {'all_TRAIN': 21878, 'no_public_anchor': 1221, 'multiple_public_polygons': 13963, 'public_text_eligible': 21, 'GT_defined_angular_labels': 21, 'engineering_axis_halfplane_matches': 6, 'engineering_axis_8bin_matches': 1}, 'per_predicate': {'3': 8, '4': 13}, 'public_eligibility_fraction': 0.0009598683609105037, 'engineering_halfplane_agreement': 0.2857142857142857, 'engineering_8bin_agreement': 0.047619047619047616, 'geographic_CRS_verified': False, 'GT_agreement_never_used_for_selection': True, 'source_sha256': '8a747e3cfa3e52c166009d09c34fe3f6c3947ff77685824433a84a840a5aebaf', 'public_catalog_sha256': 'f71ef8117f34434b77b9f608a6232a2efd48553db0d922974d91e17babd7bf74'}。只有21条自动资格、N/S8/13、E/W词样本0；5轮重复不是105条独立监督。GT半平面/8bin一致率只在原工程轴约定下统计，不能用它选样本或翻转坐标。TIF CRS=None、无标签，无法声明+Y是地理北；道路全Polygon顶点均值也不等于局部道路参照。Angle Loss训练真实GT地图方位，并不硬编码north=+Y标签。

CPU412项通过；真实TRAIN纯Angular对Geometry和Relation非零梯度，Predicate Embedding第二步收到梯度，首步0来自原score零初始化。随机4batch Smoke恰好无资格样本，另用按公开文本选择的2例验证Angular反传，未按GT筛选。后续每组2条TRAIN真实闭环只验证接口、不报告SR。

## TRAIN Birmingham拟合诊断（非独立Dev）

Birmingham7619条包含在TRAIN，GT仅损失/事后评价。概率质量是主Belief归入真实8方向类别的诱导概率，不是像素或实例识别。

|方法|R20|NE(m)|角度样本N|角度8bin准确率（比例）|GT方向概率|
|---|---:|---:|---:|---:|---:|
|matched_full_text|66.308%|26.6059|17|0.4117647058823529|0.2796|
|predicate|66.374%|26.5947|17|0.47058823529411764|0.2816|
|angle01|66.269%|26.6642|17|0.47058823529411764|0.2856|
|angle03|66.400%|26.6006|17|0.47058823529411764|0.2895|

A对匹配全文：{'fixes': 14, 'damage': 9, 'net': 5}；B/C对A：{'angle01': {'fixes': 16, 'damage': 24, 'net': -8}, 'angle03': {'fixes': 26, 'damage': 24, 'net': 2}}。多Polygon、自动间接链、无Anchor等重叠分层见TRAIN_location_direction.json；未判定直接不等于间接，链标签是公开词法代理而非人工语义真值。

## 全2470/2470 Val-Seen真实闭环

三个固定第5轮模型分别真实闭环，原起点/完整指令/Controller/Stop0.3/1mm守卫/Horizon20/20m均不变；RGB更新关闭以隔离本轮语言/几何方法。无Anchor/无资格/错误任务全部入分母，无GT角度在线使用，不访问Test。

主对照是已完成、同公开绑定重训的全文47.37%（9ac51ee2），不是早期旧训练绑定权重45.75%。前者使A与无Predicate模型的区别更纯；旧45.75%只保留历史上下文。Base/A/B/C所有Episode、起点、原文和公开Polygon逐条核对一致。

|方法|SR|OSR|SPL|NE(m)|nDTW|
|---|---:|---:|---:|---:|---:|
|matched_full_text|47.368%|54.008%|0.4598|34.2257|0.2948|
|predicate|47.409%|54.170%|0.4601|34.1458|0.2948|
|angle01|47.409%|54.089%|0.4601|34.1606|0.2949|
|angle03|47.449%|54.211%|0.4604|34.1506|0.2949|

配对与物理目标分组95% bootstrap（区间为比例，乘100为百分点）：{'matched_full_text_to_predicate': {'fixes': 4, 'damage': 3, 'net': 1, 'SR_delta95CI': [-0.0016207619672053604, 0.0024330900243309003]}, 'predicate_to_angle01': {'fixes': 7, 'damage': 7, 'net': 0, 'SR_delta95CI': [-0.0028606749265959186, 0.002839785404757514]}, 'predicate_to_angle03': {'fixes': 8, 'damage': 7, 'net': 1, 'SR_delta95CI': [-0.0028537359501485064, 0.0036600244001626678]}, 'matched_full_text_to_angle01': {'fixes': 6, 'damage': 5, 'net': 1, 'SR_delta95CI': [-0.0024164317358034634, 0.0032052890405459646]}, 'matched_full_text_to_angle03': {'fixes': 7, 'damage': 5, 'net': 2, 'SR_delta95CI': [-0.0020317150804584033, 0.004037182466413133]}}。单训练种子，不能将Episode CI解释为多种子稳定性。A−全文用于Predicate效果；B/C−A用于Angular新增效果，不选择Val最优α再假装独立确认。

完整Checkpoint/原始逐步轨迹/大体积离线字段保留/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_angular_s42_20261106；代码、配置、曲线、全部配对CSV和真实热图/成功失败轨迹提交GitHub。冻结SHA见FROZEN_BEFORE_VAL.json，所有未训练模块逐张量与Original V2一致。未修改main，不覆盖旧权重，不更改阈值，不训练目标外观或新控制器。
