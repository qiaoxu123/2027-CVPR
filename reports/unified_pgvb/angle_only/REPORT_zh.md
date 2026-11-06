# PR25：8方向方位角预测与Belief融合

## 设置与输入安全

全部21878条Revised TRAIN、seed42、3轮，冻结原Unified V2 Geometry/Relation/PGVB、BERT和原导航控制器；只训练231817个新增参数。Loss=Goal CE+0.5 Direction CE+0.05 KL，FP32且TF32关闭。3轮每轮覆盖21878条，实际最后训练记录耗时514.43秒，训练内GPU步骤均值32.174ms（不含DataLoader等待），峰值GPU分配0.747GiB。未修改原权重。本轮按PR25指定77cd5560旧全文Checkpoint为基座，并非上一轮SR47.37%的新全文权重，不能混用基线。

新增头只接收冻结全文/公开名称嵌入、公开候选方位角、有效Anchor Mask和原Belief；无dx/dy、距离、距离桶或GT目标位置。atan2在候选公开几何预处理/训练标签中计算，坐标差和模长不进入新增头。原冻结主模型仍保留原几何输入，本实验不是全模型去除距离。方向为地图+X=0度、逆时针8类，未证明地理北；1m内重合Anchor没有有效方向标签。

修复原PR按目标ID索引processed_descriptions读地标列表的路径：TRAIN及推理均从原文匹配完整公开目录，使用同一public_text_record/full_text，原私有目标ID和GT选锚不进入模型。GT只在forward后构造方向/Goal损失和TRAIN离线Oracle。无Anchor全部回退原Belief并入分母。Anchor采用原特征约定的完整Polygon顶点均值（不是面积质心或GT最近边界）；同名多Polygon独立保留并平均角度相容性，超容量不按GT选择。与已有45.75%基线核对同Checkpoint、公开绑定、起点及控制协议，全部初始角度修正前logits最大差值0.0，复用其全部真实轨迹。

## TRAIN内定位诊断（非独立Dev）

Birmingham7619条也参与训练，只能作为拟合诊断。Oracle无闭环开关，不在Val上计算GT方向。

|方法|R20|NE(m)|
|---|---:|---:|
|原全文冻结基线|64.195%|33.8403|
|预测8方向|64.155%|33.8358|
|Oracle8方向（仅TRAIN离线）|64.956%|33.3591|

预测方向分类准确率41.125%，有效角度标签17381，有角度支持任务7069。预测定位纠正6、破坏9，净-3。Oracle只评估相同已训练有界融合公式中的GT8方向贡献，不是任意位置Oracle上界。

同名多Anchor的事后监督歧义统计：{'with_public_anchor': 20657, 'no_public_anchor': 1221, 'anchor_queries': 69778, 'queries_with_identical_name_inputs_multiple_anchors': 7888, 'queries_with_conflicting_GT_angle_bins_for_identical_head_inputs': 6590, 'conflicting_name_groups': 10112, 'conflict_audit_GT_offline_only': True, 'TRAIN_angle_label_histogram': [9495, 9750, 10473, 7699, 9063, 7536, 8213, 7480], 'Birmingham_angle_label_histogram': [2701, 3008, 3250, 1703, 2154, 1504, 1495, 1566], 'Birmingham_majority_class_baseline': 0.1869857890800299}。同一指令和同一名称会给预测头相同输入；如果不同片段的GT方向类不同，当前头只能给出相同的混合方向分布，不能识别片段身份。该统计仅用于事后解释，不据GT选择Anchor或修正预测。

## 全2470/2470 Val-Seen真实闭环

冻结第3轮，不用Val调参；只输入预测角度，Oracle从未进入导航。保留原20m、Horizon20、Stop0.3、1mm到达守卫、Stable Goal与动作策略；RGB更新关闭，与原语言/几何基线一致。

|方法|SR|OSR|SPL|NE(m)|nDTW|
|---|---:|---:|---:|---:|---:|
|baseline|45.749%|53.279%|0.4440|37.9596|0.2888|
|predicted_angle|45.709%|53.239%|0.4437|37.9395|0.2889|

配对纠正3、破坏4、净-1。实际错误0，初始Top1变化24条，有角度支持2344条；残差最大0.072413、中位0.014053logit。OSR成功而SR失败186条。SR差值物理目标组bootstrap95%区间（比例单位，乘100为百分点）[-0.002849031849439016, 0.0020096463022508037]；一个训练种子，不宣称Unseen或多种子泛化提升。

新增闭环推理平均25.490ms、耗时619.72秒；旧基线是历史运行，不能作同一时段速度提升结论。

## 角度到Belief转换的限制

残差与原logit间隔：{'N': 2470, 'baseline_Top1_Top2_gap_median': 0.11240673065185547, 'angle_delta_range_median': 0.022014113143086433, 'Top1_provably_cannot_change_by_global_delta_range': 2140, 'posthoc_only': True}。满足gap大于delta整体范围的任务，角度残差在数学上不足以推翻原Top1；不据此修改Val上的融合系数。这仅诊断当前已训练有界融合，不证明真实角度信息没有潜力。

## 实现核查与局限

CPU专项与全测试见CPU_tests.txt；梯度审计证实新增头非零、原模型无梯度；训练后原模型逐张量完全一致。2条TRAIN真实闭环仅作接口验收，使用只暴露公开objects的catalog，无法调用目标索引接口，不评分。修复public-text模式覆盖角度Policy的问题、最终TopK/坐标/128维Context同步及角度修正热图捕获问题。普通默认导航未改变。

视觉实例身份与Polygon投影像素准确性不在本轮验证范围。多Anchor平均、粗8类和±1有界logit是原PR设计，未根据Val调强度或改结构。详见TRAIN24例Polygon/方向概率/三组Belief热图及反事实JSON。固定长度箭头仅供显示，不是输入距离或预测距离。

角度Checkpoint：/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_angle_only_s42_20261106/full/last.pt
SHA256：c30eb96ad408f07eb96bb5f02dd88ebcf4fcdd283c632a1542dfef190502182f
原模型SHA256：77cd5560a70d9ab3b1c9ec1a9f17efbd78732f8809b7f6d140620c6a9378e04b
服务器保留全部日志、Checkpoint、轨迹和逐步字段：/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_angle_only_s42_20261106。未访问Test、未合并main。

## 本轮结论与下一步

方向分类在TRAIN内为41.13%，高于同集合多数类基线18.70%，说明新增头学到部分数据相关性；这不能单独证明泛化的关系理解。12个合成方向替换诊断在工程+Y/+X约定下分别有6/12和11/12出现对应类别概率上升，地图轴尚未证明为地理北，因此不作为真实地理方向准确率。

本轮全量SR纠正3、破坏4，差值−0.0405个百分点；95%物理目标分组区间约[−0.285,+0.201]个百分点，覆盖0，无可验证净收益。NE减少约0.020m，不构成有意义的导航改善。TRAIN Oracle的+0.76个百分点R20只适用于当前已训练融合强度，不是角度信息的理论上界。

两个优先瓶颈：6590条TRAIN中，同一指令/同一名称输入对应多个片段、不同方向标签，当前方向预测头缺乏片段可辨识性；以及2140条Val-Seen原Top1-Top2差值大于整个角度残差范围，当前修正不能改变Top1。先解决公开Anchor身份/多片段语义及角度证据到Belief的有效转换，并在独立TRAIN分组验证；不依据本次Val结果调增益、不盲目加轮数，不用Oracle结果宣传导航收益。

392项CPU测试通过，全部GPU训练与闭环真实执行，原主模型张量不变，所有初始角度前基线logits与历史基线最大差值0。独立adversarial-review已调用，但服务workspace routing discovery timed out，未计为审查通过。原Checkpoint、历史报告和main均保留；本轮未访问Test。
