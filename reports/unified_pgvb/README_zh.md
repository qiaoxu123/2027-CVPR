# UnifiedPGVB-v2：几何、语言与单视觉Belief统一网络

此文档记录V2真实代码。当前完成结构整合和权重迁移工具，不代表已重新训练或取得新SR。

## 架构：只有一个负责GoalField的网络

- UnifiedPGVB.geometry：原GraphNav A3的完整语言+公共地图几何定位权重，输出28×28 logits、Top-K候选与128维稠密上下文。
- UnifiedPGVB.pgvb：唯一RGB视觉Head。SigLIP2视觉/文本骨干冻结，当前真实384×384 RGB只编码一次；利用名称、类型和合法局部Polygon投影输出地标可见证据。
- UnifiedPGVB.forward_field：内部完成几何前向、局部PGVB评分、世界网格有界融合、Top-K重排与候选上下文索引，直接返回一个完整GoalField。无可信图像证据时返回未修改的几何Belief。
- UnifiedPGVB.direction：方向变换模块权重可选迁移，尚未在实际目标闭环启用。
- UnifiedPGVB.target_text/target_score：预留目标外观Head，尚未训练，必须拒绝推理。
- Trajectory/Stop：只保留原GraphNav冻结的轨迹与停止控制头，不再产生或修改GoalField。

原Stage1模型壳只用于严格加载已有权重；之后从中迁走Geometry、保留Controller，并释放其余模块。新运行时没有单独A3前向、Matcher、History或外部PGVB融合。

## 速度与正确性约束

一个Episode的冻结Geometry结果在UnifiedPGVB内部缓存一次，下一Episode强制重置；若Geometry参与训练，则禁止缓存断梯度输出。RGB仅单次SigLIP2视觉前向，不重复裁剪编码。保留原28×28网格、候选上下文、Guarded-Blend、Stop规则、horizon和yaw，不用旧成绩冒充本版结果。

## 只迁移权重，打包统一v2 Checkpoint

~~~bash
python -m scripts.build_unified_pgvb_v2 \
  --stage1 /actual/path/to/graphnav_stage1.pt \
  --pgvb /actual/path/to/local_pgvb_best.pt \
  --direction /actual/path/to/direction_real.pt \
  --output GraphNav/runs/unified_pgvb_v2.pt
~~~

方向权重可暂时省略（此时禁用方向头）；原始Stage1、PGVB与Direction权重不覆盖。输出格式unified-pgvb-v2，内含Geometry权重、来源SHA256、jointly_trained=False、target_ready=False。V2运行时严格拒绝旧v1权重与不匹配的Stage1 Controller来源。

## CPU验证与真实GPU Smoke

~~~bash
python -m pytest -q \
  tests/test_unified_instruction.py \
  tests/test_unified_pgvb.py \
  tests/test_unified_pgvb_v2.py \
  tests/test_pgvb_navigation_runtime.py

python -m scripts.run_unified_pgvb_landmark_closedloop \
  --selection /actual/path/to/frozen64_selection \
  --checkpoint /actual/path/to/graphnav_stage1.pt \
  --unified-checkpoint GraphNav/runs/unified_pgvb_v2.pt \
  --sbf-source /actual/path/to/SBFNav \
  --config /actual/path/to/config.yaml \
  --data-root /actual/path/to/CityNav \
  --output GraphNav/runs/unified_pgvb_v2_landmark64 \
  --max-new-tasks 2
~~~

Smoke通过后追加--resume完成同一64条固定诊断。需要记录单次编码次数、端到端每决策耗时、Geometry与最终Belief差、Goal跳变、Stop以及任务SR/OSR/NE。64条是公开地标Polygon任务，不等于CityNav原始Target SR。不得复制历史E4的成绩作为新版本测试结果。

## 语言与训练边界

现有GT-free结构化Parser支持简单具名地标导航（如Navigate to The Food Warehouse.）及复杂指令的离线关系分析；复杂指令尚不可进入未训练目标Head，不能将参考地标偷换成目标。

当前未完成：完整TRAIN复训、所有地标的完整SR、真实方向关系Belief导航与目标外观特征训练。建议随后基于统一V2网络训练PGVB，冻结Geometry和控制器以加快训练，确认地标和方向关系后再按需联合微调Geometry。始终禁止真实目标GT、未来观察或私有实例身份进入推理。

历史E2/E4代码及Checkpoint保持不变，V2独立入口不直接改main。
