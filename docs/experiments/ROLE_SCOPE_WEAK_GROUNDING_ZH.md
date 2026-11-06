# CityNav 下一阶段：关系作用域与中间实体绑定（短期实验）

## Goal

**在不增加模型容量和视觉模态的条件下，检验语言中“目标→中间实体→具名地标”的显式绑定监督，能否提升地图留出上的8方向预测，尤其是间接关系子集。**

上一轮TRAIN确认集：旧头13.72%、普通对象匹配38.37%、关系网络39.49%。关系链净收益+1.13个百分点，95%区间跨0；仅目标短语和去轮廓/拓扑并未更差。这里不假设增加关系深度自然带来收益。

## 最小方法

保持上一轮完整指令、公开Polygon候选、边关系、8方向角度监督和冻结BERT不变。对原关系网络只新增可选逐跳公共对象分布输出，默认推理行为和权重格式不变。

公开弱监督只由现有GT-free语言解析和公开Polygon目录产生：
- 目标、地标、直接/间接角色沿路径从Anchor向目标反向遍历；
- 如果中间实体在指令中直接具名且绑定到公开Polygon，监督对应**公开候选集合**，不挑选单一GT实例；
- 如果只出现普通实体（如“停车场”“汽车”），监督**公开场景中所有该类别候选**的概率质量，不能制造真实实例标签；
- 关系边主客体、作用域或同长度路径冲突时不提供辅助监督，原角度训练保持不变；
- 一个地标和目标实体不能用目标GT坐标、目标ID筛选邻域或赋予角色标签；输出为弱标签，不是人工标注。

训练目标：`L = L_angle + lambda_role * L_public_weak_role`。目标GT角度仍仅用于`L_angle`。弱监督使用候选集合边际概率和query等权，按实体可信度加权。

## 三组同容量、同初始权重消融

| 组 | 新增辅助监督 | 意义 |
|---|---|---|
| A angle_only | 无 | 上一轮关系网络的角度训练方式 |
| B weak_intermediate | **只监督中间实体** | 检查间接关系链是否真的获益 |
| C weak_role | 中间实体 + 宽泛目标类别 | 检查总体角色监督及目标类型先验增益 |

三组均从随机初始化重训，不能拿不同确认集/旧epoch的最高分冒充公平对比。先打印公开弱标签覆盖率；若中间实体覆盖不足，必须如实报告，不应靠加大网络或GT推断填补。

## 执行命令

在原先有真实数据与GPU的机器、仓库根目录下：

```bash
python -m pytest -q tests/test_role_scope_supervision.py tests/test_relation_bearing_network.py

RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106
OUT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/role_scope_weak_s42_20261106

python -m scripts.prepare_role_scope_supervision --run "$RUN" --output "$OUT"
python -m scripts.train_role_scope_bearing --run "$RUN" --output "$OUT" \
  --epochs 12 --batch-size 16 --lr 3e-4 --role-weight 0.2
```

在第一次正式训练前，也可在**新目录**用`--max-updates 3`做GPU smoke；不要把smoke数值当实验结果。正式训练输出各组 `best.pt`、`last.pt`、`history.json`，以及 `DEVELOPMENT_REPORT.json`，展示query-macro Top-1、anchor Top-1/Top-2、圆周MAE、NLL和逐query纠正/破坏。

## 解释约束

1. **原来四张confirmation地图已看过结果**，不得再拿这些地图反复调参并称“新独立确认”。当前代码只读取既有TRAIN fit/dev；要做独立结论需重新预注册未使用过的划分或外部数据，并在冻结后评测。
2. 直接/间接自动分类仍是弱解析，尚无严格人工金标准。需要单独抽样人工检查路径的主客体、指代与否定范围。
3. 若B未明显优于A，说明当前可用中间监督、候选覆盖或关系推理不足，不能声称复杂关系链奏效；若只有C改善而B不改善，要警惕收益仅来自目标类别先验。
4. 现阶段不加入RGB、不重新训练GeomBelief、不进行导航闭环；不能把方向Top-1和导航SR混为一谈。

## 代码文件

- `graphnav/relation_scope_supervision.py`：GT-free弱标签及边际集合loss；
- `graphnav/relation_bearing_network.py`：增加`emit_steps=True`逐跳分布接口，默认不改变原行为；
- `scripts/prepare_role_scope_supervision.py`：只分析公开解析/场景和TRAIN fit/dev；
- `scripts/train_role_scope_bearing.py`：同配置三组训练、开发集配对评价和Checkpoint；
- `tests/test_role_scope_supervision.py`：作用域、候选集合、GT隔离、梯度与原模型一致性测试。
