# Unified V2 + Structured Instruction Graph: 第一阶段可运行实验

**不要重新训练已有完整指令基线，也不要替换完整指令。**
本方案是原 Unified V2 完整指令 Geometry+Relation Belief 上的
一个可开关两层图编码器残差（Graph Belief Residual），
不是 PR #22 的独立小模型或 PR #23 的精简指令替换实验。

## 架构

原始完整指令 -> 原冻结 BERT tokens + 原 Unified V2 Geometry+Relation
        -> 原始 Goal Belief [B,784] + dense_context [B,784,128]

与此同时：

原始指令 + 公开 CityRefer 名称目录（不读 GT）
        -> 现有 spaCy/structured_navigation 自动解析提案
        -> 节点：target（类别/颜色/目标特征）、intermediate、public landmark
        -> 有向边：predicate、reference frame（含未知）、direct/indirect paths
        -> 每个节点/边分别经冻结 BERT 编码，绝非整句 CLS
        -> 2 层双向 Typed Graph Message Passing（保持边方向）
        -> 与目标节点、路径终点地标和 10D Polygon 相对位置融合
        -> bounded / zero-initialized candidate-specific residual
        -> Goal Belief = 原完整指令预测 + 图修正

目标/中间实体如果没有公开真实几何，就保留 unknown geometry，
**不能用目标 GT 为它们补位置**。图边的方向语义来自解析器提案，
不等同经人工验证的空间关系。解析失败、无路径或路径终点无法绑定
已有公共地标 Polygon，则残差严格为零，沿用完整指令预测。

同义改写的图结构、节点属性和关系一致时，编码器对**任意节点重编号**
不敏感；但如果解析器对两种语序产生不同图，就不能声称有语序不变性。

## 本阶段训练边界

- 复用已经训练完成的原 Unified V2 Full Relation Checkpoint，不重训它。
- 冻结核心 Geometry、Relation、BERT、PGVB 和 Controller/Stop。
- 只训练新的两层 Graph Encoder + Polygon 候选残差头；
  损失为真实训练目标位置 CE + 0.05 KL（保持原始 Belief）；
  原有 GT 在网络 forward 之后用于训练 Loss。
- 所有 21,878 条 Revised TRAIN 仍进入优化统计；
  但只有存在可信可绑定解析路径的任务会有新增头梯度。
- Birmingham 属于这 21,878 条 TRAIN 中的一部分，
  此处报告 R@20m、NE 是训练内诊断，不是独立验证。
- 暂不运行 Val/Test/2470 闭环；先检查图解析覆盖、模型是否使用关系边、
  node order invariance 和基础定位变化。需要增益后再实现正确的
  UnifiedPGVBFullRelationPolicy runtime 融合并跑全量 Val-Seen。

### 本地命令

下面路径沿用现有部署，只要实际文件存在可直接执行。
若服务器代码路径不同，应以现存复现脚本为准，不覆盖旧实验目录。

~~~bash
cd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python
RUN=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_full_relation_s42_20261105
OBJECTS=/home/20261024_1/Workspace/DATA/sbfnav-data/cityrefer_original/objects.json
OUT=/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_graph_belief_s42_20261106

# 0. 验证 PR24 两层图模型 CPU 测试，不运行旧 PR19 需要人工确认标签的训练脚本
"$PY" -m pytest -q tests/test_unified_instruction_graph_belief.py

# 1. 一次性生成全部弱图提案（原始完整指令 + PUBLIC map catalog）
# 首次执行前检查：本地 python 可 import spacy / en_core_web_sm
"$PY" -u -m scripts.prepare_unified_instruction_graphs \
  --train-jsonl "$RUN/train_revised.jsonl" \
  --public-objects "$OBJECTS" \
  --output "$OUT/weak_graphs.jsonl"

# 2. 小范围 CUDA Smoke；此结果不代表全量拟合
"$PY" -u -m scripts.train_unified_instruction_graph_belief \
  --train-jsonl "$RUN/train_revised.jsonl" \
  --graph-jsonl "$OUT/weak_graphs.jsonl" \
  --existing-full-checkpoint "$RUN/core/last.pt" \
  --output "$OUT/smoke" \
  --epochs 1 --batch-size 8 --seed 42 --max-batches 12

# 3. 真正的完整 TRAIN：只新增图分支，原模型基线一律不重训
"$PY" -u -m scripts.train_unified_instruction_graph_belief \
  --train-jsonl "$RUN/train_revised.jsonl" \
  --graph-jsonl "$OUT/weak_graphs.jsonl" \
  --existing-full-checkpoint "$RUN/core/last.pt" \
  --output "$OUT/full" \
  --epochs 3 --batch-size 8 --seed 42
~~~

断点：仅支持完成整个 epoch 后通过完全相同参数的 --resume 继续，
manifest SHA 检查原训练文件、图文件、代码和基线权重。
严禁在旧训练输出上覆盖 resume。

## 重点验收（不要为了报喜而忽略）

1. 抽查指令如蓝车在停车场后方：目标名称、属性、停车场中间节点、
   地标 One Stop 和两条有向边均存在。没有把停车场的 behind
   错当作汽车直接 behind。解析失败不能伪造图关系。
2. 全部 21,878 条输入不丢弃，且原 Full Relation 权重、
   原始全文 tokens、原 Polygon 来源和最终候选网格不变。
3. 报告总解析成功率、直接/间接路径支持率、图残差实际生效比例；
   包括无法绑定的实体和无 GT 几何的 intermediate。
4. 通过节点重编号测试：相同图不同序号应得到近似相同 Belief。
   进一步用不同语序的真实句子测试 *解析出来的图是否一致*。
5. 对同一 Polygon 的关系词做 controlled swap，
   观察网络图残差/Belief 是否变化；north/south 的物理方向
   必须先确认地图 CRS，不能把工程 +Y 约定冒充地理真值。
6. 对全 Birmingham TRAIN 诊断报告基础/图增强 R@20m、NE、
   修复/破坏，并分直接、间接、无路径分组；若图引入较多破坏，
   先检查解析准确性，不以更多 Epoch/更大模型掩盖问题。
7. 当前第一阶段没有 Graph 的在线 runtime 闭环接入代码，
   不能直接用原 Unified Policy 声称新图模型的导航 SR。
   需要先对齐实际在线 catalog/Polygon/episode 信息，再实现
   残差注入 GoalField 的 Top-K、context 和 Stop 链路。

## 已知风险

- PR19 的规则+spaCy parser 目前未经人工真值验证，最多是
  weak proposal；真正瓶颈大概率在 target/intermediate/path scope。
- 有些地图同名 Polygon 没有和旧源 selected-reference ID 对齐，
  对这些任务维持原 Belief，没有进行 GT 猜测选 Polygon。
- 当地标 Polygon 形状过于复杂或路网拓扑缺失时，自动解析不能
  可靠替代真实目标定位真值。
- 两层图消息传递可编码目标 -> 区域 -> 地标的两跳，但
  不能自动推导未知 intermediate 的物理坐标；
  这需要后续几何弱监督或可靠视觉观测，不应从最终 Goal Loss 猜。
