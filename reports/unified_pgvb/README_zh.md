# UnifiedPGVB：单视觉分支整合（实验版本）

**目标：只保留一个RGB视觉前向和一个PGVB中心视觉网络。** 不再把原GraphNav视觉Matcher、PGVB和后续Target Visual Head作为并行视觉网络。

## 架构

- 当前真实384×384 RGB → **一次冻结SigLIP2视觉编码**（同时提供共享文本表征）。
- **UnifiedPGVB** 是唯一可训练视觉网络：内部含PGVB地标评分、方向关系变换和预留的目标外观评分头，使用同一PGVB视觉投影。
- 冻结A3仅给出全局几何GoalField；UnifiedPGVB提供局部可见证据，采用保守±0.5 logit更新。
- 原Trajectory/Stop和Guarded-Blend控制器保留。1mm到达守卫不变。
- 在新的单视觉Policy实例中，严格加载原GraphNav权重后，**删除旧Matcher、History、Visual Gain及A3旧RGB评分层**，不会再执行它们；历史实验入口和原始权重文件不改。
- 这是一套统一网络与Checkpoint容器，不等于已联合训练。Direction可从现有32参数权重初始化；Target Head未训练、默认强制拒绝推理。

## 权重打包

在服务器工程根目录运行（路径从既有实验日志中确认，不猜测）：

```bash
python -m scripts.build_unified_pgvb_checkpoint \
  --pgvb /actual/path/to/local_anchor_pgvb_best.pt \
  --direction /actual/path/to/direction_real.pt \
  --output GraphNav/runs/unified_pgvb_v1.pt

python -m pytest -q tests/test_unified_pgvb.py tests/test_pgvb_navigation_runtime.py
```

如果暂不验证方向变换，省略 `--direction`，统一网络会拒绝无权重的方向推理。打包脚本不会修改原PGVB、原Stage1、原方向Checkpoint。输出标记 `jointly_trained=False`、`target_ready=False`。

## 单视觉闭环Smoke

```bash
python -m scripts.run_unified_pgvb_landmark_closedloop \
  --selection /actual/path/to/frozen64_task_selection \
  --checkpoint /actual/path/to/original_stage1.pt \
  --unified-checkpoint GraphNav/runs/unified_pgvb_v1.pt \
  --sbf-source /actual/path/to/SBF_SOURCE \
  --config /actual/path/to/original_config.yaml \
  --data-root /actual/path/to/CityNav \
  --output GraphNav/runs/unified_pgvb_landmark64_v1 \
  --max-new-tasks 2
```

Smoke之后通过同一命令加 `--resume` 执行剩余64条。该入口仍是**以公开具名地标为目标**的诊断任务，不是原始CityNav目标SR。由于旧视觉分支已撤掉，新策略不能直接复用E4的62/64结果；必须重新运行。

## 后续扩展只在同一网络内进行

1. 将完整指令解析成具名Anchor、直接空间关系和目标外观；不确定参考系时弃权。
2. `UnifiedPGVB.direction`输出空间目标先验（非RGB编码器）。
3. `UnifiedPGVB.target_text / target_score`使用同一SigLIP2图像特征进行目标外观评分，**训练、验证通过后才允许启用**。
4. 两者在同一世界坐标网格上融合成一个Target Belief。不可见时保留几何先验，不强行将未知格置为负样本。
5. 仅用原始目标GT训练监督及离线评测；闭环前先验证不同方向、目标可见性与视觉增益。

**当前完成的是架构整理、权重打包和具名地标单视觉运行路径；未完成Target Head的监督训练与原始CityNav闭环实验。**


## 统一语言解析接入（新增）

当前统一语言接口在 graphnav/unified_instruction.py，复用仓库已有
graphnav/structured_navigation.py 的复杂关系解析输出，不新增LLM或视觉骨干。

### 两种指令的不同处理

- 具名目标：Navigate to The Food Warehouse.
  - 无需加载spaCy，按完整指令识别 navigate_to_landmark；
  - 仅依据公共地图作精确或带类型后缀的地标名称绑定，保留所有同名合法Polygon片段；
  - 验证解析出的目标和episode.reference_names指向同一个具名地标；
  - 统一网络直接通过 UnifiedPGVB.forward_structured 接受结构化任务，并使用唯一PGVB视觉Head给出地标证据。
- 目标相对地标：Find the white building north of The Food Warehouse.
  - 可选加载已有spaCy/structured_navigation解析器，保留目标类型、颜色、原文span、直接/间接关系、方向、作用域和参考系歧义；
  - 语义/地理方向、实体绑定和目标视觉识别尚未确认，解析结果 approved_for_inference=false；**不得直接导航到The Food Warehouse作为替代**；
  - 原始指令保留，分析时可以查看解析失败原因；不把自动解析当作人类Gold。

### 批量审查语言结果

输入JSONL，每行只提供 map_name、instruction，可选 episode、instruction_index。
公共Catalog为CityRefer objects.json结构。

~~~bash
python -m scripts.audit_unified_language \
  --instructions /path/to/public_instructions.jsonl \
  --catalog /path/to/cityrefer/objects.json \
  --output GraphNav/runs/unified_language_audit.jsonl

# 对复杂关系使用已有spaCy解析器（需要en_core_web_sm）
python -m scripts.audit_unified_language \
  --instructions /path/to/public_instructions.jsonl \
  --catalog /path/to/cityrefer/objects.json \
  --output GraphNav/runs/unified_language_rich_audit.jsonl \
  --rich

python -m pytest -q \
  tests/test_unified_instruction.py \
  tests/test_unified_pgvb.py \
  tests/test_pgvb_navigation_runtime.py \
  tests/test_structured_navigation.py
~~~

普通PGVB具名任务运行仍使用 scripts.run_unified_pgvb_landmark_closedloop
现有入口；它现在显式拒绝复杂/不可靠指令和错误的reference_names，防止
把参考地标误作真正目标；已验证的简单指令继续使用之前的PGVB权重。

此分支实现的是**语言解析到统一视觉网络的可靠接线和边界保护**，
不是已训练的复杂Target Belief；方向、目标属性和关系融合必须在
人工核验语义标签与监督训练后再打开。
