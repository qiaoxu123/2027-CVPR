# RG-GBF：Relation-Guided Gated Belief Fusion（独立实验版本）

## 本次真正新增了什么
不是只冻结 AerialVG 做推理；**冻结的 AerialVG Native Query 是视觉底座**，
新建的 `RelationAwareQueryRanker` 会参与反向传播，学习语句中的**目标相对于具名地标的方向、距离、空间匹配**，
并输出对原 Query 评分的可训练修正。关系匹配结果再交给新训练的 Gate 生成/决定是否采纳 Visual Belief。

本版本**没有复制官方 AerialVG 的完整 Relation Transformer**：
采用其 **target-reference relation attention** 思想，并用 CityRefer
全多边形/已核验道路中心线和世界坐标取代原方法图像内物体框的几何关系，
更符合 CityNav 的城市级目标定位。不要将其描述为官方 AerialVG 的复现。

## 代码
- `graphnav/native_relation_grounding.py`：从 A0 使用的同一静态地图图谱构建公开关系特征；
  候选 Query × 具名参照地标 Relation Cross-Attention；有界关系残差；20m弱监督排序 loss
- `scripts/train_native_relational_grounding.py`：
  TRAIN 中按 map+instruction group 拆为 fit/calibration，两阶段训练：关系排序器、Gate；
  冻结 A0/原 Native；仅 TRAIN 校准阈值；最后只做一次内部 Dev 诊断
- `tests/test_native_relation_grounding.py`：基础CPU安全测试
- 继承 `graphnav/gated_belief_fusion.py`：全局视觉概率假设与回退

## 标注/边界
- 关系解析由现有 `semantic_graph` 的公开指令、具名地标多边形生成。
  同名多实例、不匹配的参照物**不使用 GT 选中正确实例**，会禁用该关系。
- 当前候选“目标”仍仅有半径20m的 **spatial-utility 弱标签**，
  不等于已确认目标物体 identity。无位置正例时跳过排序 loss，
  **禁止把无20m候选写为目标不可见的身份监督**。
- relation head 在初始化时严格输出原生 Native 评分，
  无明确关系的指令严格回退 Native。
- Gate 无可靠支持时回退 A0，threshold=1.0 表示绝对不更新。
- Gate 分数从新关系排序重新训练；不复用旧 Gate checkpoint，
  避免评分分布改变后未经校准便投入导航。
- `pilot.pt` 将 `admitted_for_runtime=False` 写入，正式导航不接入。
- 该 Commit **无训练结果、无正式SR提升声明**。
- 现有内部 Dev 635状态只是探索性诊断；不可当作官方独立确认。

## 本地测试（先 smoke）
切换新分支后：
```sh
python -m pytest -q tests/test_native_relation_grounding.py tests/test_native_gated_belief_fusion.py

python -m scripts.train_native_relational_grounding \
  --train Ablations/GraphNav/visual-residual-20261013/aerialvg_native_training_train256_20261101 \
  --dev Ablations/GraphNav/visual-residual-20261013/aerialvg_native_training_dev128_20261101 \
  --stage1-cache GraphNav/data/rebuilt_stage1_20261025/frames \
  --native-checkpoint Ablations/GraphNav/visual-residual-20261013/aerialvg_native_plain_sigma8_s0_20261101/best.pt \
  --output Ablations/GraphNav/visual-residual-20261013/rg_gbf_smoke_s0_20261101 \
  --epochs 1 --gate-epochs 1 --seed 0 --device cuda
```
路径按实际工作区根目录调整。确认无异常后用**新输出目录**改为
`--epochs 4 --gate-epochs 3`。再做 seed1、seed2，禁止覆盖原实验。

## 对照必须报告
在相同635内部Dev状态：
1. 冻结 A0、Native旧Residual、`relation_residual`、`relation_gate` 的R@1、NE、纠正/误改/净收益
2. 有具名、无具名、同名多义、目标20m位置覆盖、A0原错等子集指标
3. **候选排序**：固定Top15前后正确率（而非只有最终Belief R@1）
4. 关系真实/关闭/空间关系打乱消融（固定视觉与语言输入），验证关系特征确实有效
5. Null概率分层、Gate接受率、拒绝可纠错/接受误改、Train校准阈值
6. 参数量、实际CPU/GPU耗时、单步新增推理开销
7. 数据/Checkpoint哈希、Train/Dev组不交叉；不得用Dev改超参

如 `relation_present` 比例太低（例如几乎所有指令都没有唯一具名参照），
要停下检查 parser 与地图别名，不可把没有关系的样本反复训练并宣称
关系效果。若更新后仍不优于Native +7/635净纠错，停止推广。

**正式 Val-Seen 仍暂停；不运行 Val-Unseen/Test；不更新原权重。**
