# UnifiedPGVB-v2：几何、语言与单视觉Belief统一网络

此文档记录V2真实代码。当前完成结构整合和权重迁移工具，不代表已重新训练或取得新SR。

## 架构：只有一个负责GoalField的网络

- UnifiedPGVB.geometry：原GraphNav A3的完整语言+公共地图几何定位权重，输出28×28 logits、Top-K候选与128维稠密上下文。
- UnifiedPGVB.pgvb：唯一RGB视觉Head。SigLIP2视觉/文本骨干冻结，当前真实384×384 RGB只编码一次；利用名称、类型和合法局部Polygon投影输出地标可见证据。
- UnifiedPGVB.forward_field：内部完成几何前向、局部PGVB评分、世界网格有界融合、Top-K重排与候选上下文索引，直接返回一个完整GoalField。无可信图像证据时返回未修改的几何Belief。
- UnifiedPGVB.joint_logits/joint_context：内部额外提供几何128维上下文与PGVB视觉证据联合条件化的可训练Belief/Context细化层。最终层零初始化，joint_fusion_ready=False默认禁用，确保未训练时与旧有界融合输出严格一致；未来需监督训练、独立验证后再开放。
- UnifiedPGVB.direction：方向变换模块权重可选迁移，尚未在实际目标闭环启用。
- UnifiedPGVB.target_text/target_score：预留目标外观Head，尚未训练，必须拒绝推理。
- Trajectory/Stop：只保留原GraphNav冻结的轨迹与停止控制头，不再产生或修改GoalField。

V2运行时**完全不实例化旧GraphNavStage1网络，也不继承原PGVB策略类**：直接由统一Checkpoint构建UnifiedPGVB（含A3 Geometry），从原Stage1文件只提取冻结Trajectory/Stop控制头和协议指纹。共享RGB预处理是无额外可训练参数的工具函数，不是第二套视觉网络。新运行时没有单独A3前向、Matcher、History或外部PGVB融合。

## 主方案补充：公开地标语言绑定与多候选解码（2026-11-07）

**设计结论：建议纳入主方案的 Geometry 前置流程；当前只是方案文档更新，不代表 `main` 的默认 Runtime 已切换。** 评估依据为 `experiment/v2-top1-public-binding-20261107` 的 [阶段报告](https://github.com/qiaoxu123/2027-CVPR/blob/26a71ceabe3c50906142817e51bf9dcfd85e963a/reports/unified_pgvb/top1_public_binding/PROGRESS_20261107_zh.md) 和公开名称解析代码（源 Commit `26a71cea`）。

**方法定位：** 将 *Public Landmark Mention Resolution* 作为原有 Geometry/GGH 模块的**确定性输入适配层**，不增加独立的可训练网络，更不将其写为第三项核心创新。完整原始英文指令仍输入冻结 BERT；适配层仅把指令中出现的公开具名地标与当前场景公开 CityRefer 名称和 Polygon 建立绑定，原有候选—地标相对几何、空间规则、全局 Goal Belief 继续负责目标区域推断。

- 使用原名称、公开别名以及保守变体：场景内唯一的连续简称（最多5词）、冒号前服务商前缀去除、末词单复数归一。仅在同一场景公共目录中唯一指向**同一规范名称**时采用生成别名；冲突、通用词和重名不强行选一个。规范名称对应多个 Polygon 时全部保留，不能借助目标位置筛选实例。
- 原始指令的实际 mention span 作为关系解析可用别名保留，不用规范化字符串替换或改写完整指令。绑定的对象是**参考地标**而不是导航目标，避免把指令中的「附近地标」偷换为「目标自身」。
- Geometry 输出完整28×28全局空间分布；研究中的局部5×5 / 温度0.5 soft decoder 用于初始 Top-1 位置解码，20m 空间 NMS 用于产生可分散的 Top-K 候选。**Soft Top-1 与按原始 logits 排序的 NMS 候选不一定严格嵌套**，不得把二者 Recall 差值当作准确的可重排上界。NMS 的20m表示空间去重，不证明不同物理实例。
- 后续视觉分支只在真实当前/历史可观测 RGB/高度代理区域对候选做有界更新；无证据区域不加负视觉修正。导航目标 GT、目标 ID、未来轨迹与目标裁剪不得进入绑定、Geometry、视点选择或推理；训练标签仅在 loss 中使用。

**已获得的离线证据（不是闭环结果）：** 在 HETT Refined CityNav 完整 Val-Seen 2470/2470、原始V2 Geometry Checkpoint、新名称解析与固定解码配置下，初始 R@20m=**1251/2470（50.65%）**、NE=**29.66m**、空间 NMS Recall@5=**80.08%**。TRAIN 的 Cambridge 开发组与 Birmingham 确认组均有配对收益，但此前49.96%的最好组合采用不同检查点/解析器，不能把两者差额单独归因于名称变体。其他拟合权重在 R@20、NE、NMS@5 上没有全面占优，故现阶段保留原始 Checkpoint 作为这条研究配置的 Top-1 对照，**不自动覆盖主线权重**。

**合入运行时之前的验证门槛：** 使用完全相同的 Checkpoint、局部解码、2470任务和公共地图做旧/新解析器的逐任务配对消融；确认每个地标命中的指代、碰撞拒绝与无锚点回退；再用固定配置完整重跑闭环 SR/OSR/SPL/NE 与误改样本。Val-Seen 只能用于已冻结配置评测，不以结果反向调词典或阈值。当前默认 Runtime/Checkpoint、训练和视觉模块均保持原状。

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
