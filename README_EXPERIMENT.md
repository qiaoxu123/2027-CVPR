# 原 V1 全量 384 网格与视觉残差改进

这是 GraphNav-V1 冻结模型的独立消融 worktree，不替换主结果。

`full_resolution_384/frozen_resolution.json` 在完整高分辨率评测前冻结原 C Epoch1 权重、384 网格、原控制器、stop=0.3；三个 split 均为完整样本。只读共享 DATA / 原权重，无大型复制。

ROI 优化保留真正的全局 147456 个候选：支持像素统计采用相同凸多边形栅格的整数计数，重复 patch mask 复用相同池化结果，仅处理有效尺度，半精度特征留在 GPU。旧 28/384 初始 Top16 重放匹配；支持 mask 与质量已对旧实现验证一致。高分辨率数值舍入差异、并发负载与续跑过程均保留记录；没有将显示插值冒充真实网格。

视觉残差候选仅训练新的小头，原几何模型与控制器冻结，利用实际当前/历史原生 SigLIP 向量、文本交互、几何上下文与质量置信度；Loss 用 20m 合法区域的概率质量，并限制不必要修正。分别比较中心化修正与允许实际可见负证据整体降权的非中心化修正。未观测、零向量证据均保持静态信念不变。

使用原 Train Dev 按整 map+instruction 组划分选 checkpoint；随后执行 Seen128 与完整 Seen 的 real / no-belief / zero / shuffle 对照。完整 Seen 的配对 bootstrap 区间与帮助/损害数量作为收益证据，不能用冒烟测试或极少样本声称提升。Test 不用于新残差模型选择。

执行工具：

- `scripts.run_full_resolution_parallel`：各 split 独立进程，完整 episode 续跑；固定实验配置，不做阈值选择。
- `scripts.train_calibrated_visual_residual`：全 Train 缓存训练及整组 Dev；`--uncentered` 是绝对负证据候选。
- `scripts.visual_residual_workflow`：串联第二候选和所有对照，自动更新实时状态 / 报告。依赖本地缓存的冻结 backbones。

## AerialVG 启发的下一版方案（待验证）

2026-11-01：候选级视觉关系消息原型已接入，使用 `scripts.train_calibrated_visual_residual --relation` 启用。短训练及控制验证记录见 [RELATION_VALIDATION_20261101.md](RELATION_VALIDATION_20261101.md)。以下整体研究方案仍有尚未实现的关系监督与粗到细 ROI 部分。

第二轮已增加 `--relation-attention`、命名地图轮廓关联和困难负样本排序（`--hard-rank-weight`），采用完整Train Dev选权重。同损失普通版与关系版整体R@1均为67.80%，尚无明确关系独立增益；真实RGB对shuffle出现小幅正向信号。完整记录见 [RELATION_ATTENTION_RESULTS_20261101.md](RELATION_ATTENTION_RESULTS_20261101.md)，包含配对区间、困难子集与选择集统计限制。

参考 Liu et al., *AerialVG: A Challenging Benchmark for Aerial Visual Grounding by Exploring Positional Relations*, ICCV 2025。论文指出航拍视角中同类目标外观相似、目标小且背景面积大，因此加入层级跨注意力与关系感知 grounding。这里借鉴的是“用关系消歧、用粗到细注意力聚焦”的思想；论文的检测框、Hungarian 匹配和关系矩阵监督不能直接照搬到 GraphNav 的网格导航残差。

### 现有结果给出的约束

当前 calibrated residual 的全 Seen 配对收益为 +1.336pp，95% CI [+0.162, +2.510]；absolute residual 为 +0.486pp，CI [-0.526, +1.538]。收益虽有初步信号，但幅度有限；AerialVG 改进应作为独立候选，与冻结基线和 calibrated residual 配对比较，不应改写既有结论，也不使用 Test 选择结构或超参。

### 建议结构：关系条件、候选间联合的残差头

1. **保留冻结的静态主干与控制器。** 输入仍为静态几何 logits、已观测 RGB 原生 SigLIP 特征、质量/有效像素置信度、历史证据及其真实支持范围；未观测候选仍严格零更新。
2. **指令关系编码。** 从完整 instruction token states 提取实体与关系短语（如“桥北侧的建筑”“道路交汇处附近”），编码相对方向、近远、包含/邻接等关系；优先复用现有语言几何特征，避免另造不受约束的语言模块。
3. **关系上下文消歧。** 对本时刻及历史中有观测支持的候选构造稀疏邻接边，边特征用米制 Δeast/Δnorth、距离、方位、几何边界/地标关联及时间差。小型关系注意力/GAT 只在已观测候选间传消息，输出候选对的兼容分数；将兼容分数聚合为每个目标候选的有界 delta。关系只能来自公开地图几何或真实观测，禁止 GT 目标坐标/未来观测。
4. **粗到细但不伪造分辨率。** 先在现有 384 网格/观测池化层选出高置信局部区域，再在该区域细化视觉特征与关系边；全部 logits 仍保留，区域外不因未观测而扣分。若无法从原始 patch 特征重算细尺度，则先只做多尺度 ROI 特征消融，不能将插值称为更高分辨率 grounding。
5. **有界、置信度门控更新。** `delta_i = support_i * quality_i * gate_i * clip(relation_visual_delta_i, -b, b)`；初始 gate=0，沿用 KL/概率质量保持项。当前证据与历史证据分开编码，历史需记录观测时间、位姿和来源，防止重复帧被累计成独立证据。

### 训练目标与关键对照

- 训练集只用 Train，按整 map+instruction 分组；Dev 选 checkpoint/超参，Test 完全封存。除原目标分布损失和 KL 保持项外，增加关系排序损失：对具有不同目标候选但满足同一语言关系的配对样本，监督正确目标—参照物候选组合优于外观相似的错误组合。没有可靠参照实体标注时，不生成伪关系标签，只做关系模块消融。
- 依次比较：静态冻结；现有 calibrated residual；+关系边但无关系损失；+关系损失；+粗到细 ROI；完整模型。保持数据、网格、候选、控制器、预算和随机种子配对一致。
- 除 R@1/20m 与闭环 SR/SPL/NE，还报告按“同类近邻/需方向关系/小目标/背景占比高/无关系表达”分层的结果、校准误差、实际观测覆盖率、帮助/损害 episode 数、配对 bootstrap CI、推理延迟与显存。额外做关系短语删除/打乱、邻接边打乱、RGB shuffle/zero、历史清零对照，确认收益来自真实关系与视觉证据。
- 进入主线的门槛：Dev 不退化且关系困难子集有可重复提升；完整 Seen 的配对 SR CI 下界大于 0；zero/shuffle/关系打乱均消除相应增益；完整 Unseen 只作一次性确认。若收益仅出现在离线 R@1 或小样本 Seen，不宣称导航提升。

### 迁移边界

AerialVG 是单帧框级 grounding 数据集，其分层跨注意力在本任务上的价值需由 ROI/patch 消融验证；其关系监督依赖人工目标与辅助物标注，而当前导航数据若缺少同等标注，不能声称复现论文模块或直接迁移其指标。建议先实现轻量稀疏候选关系头，不先替换 SigLIP、384 网格、静态信念或执行策略。

最终完整指标在 [RESULTS.md](RESULTS.md)，原始证据在各输出目录。未完成项见 [TODO.md](TODO.md)，不把运行中的前缀 SR 当成全量 SR。

### 官方 AerialVG 代码核验与实际复用（2026-11-01）

已下载官方仓库至 `Workspace/third_party/AerialVG`（commit `db33e1d`），严格加载官方权重完成demo整图推理及关系/HCA/文本干预。理解和原始模块测试记录位于 `Workspace/third_party/aerialvg_validation/UNDERSTANDING_AND_TESTS.md`。

本实验新增 `--official-relation`，直接复用官方K² pair-token模块和预训练权重，训练导航视觉/语言投影与零初始化门。两轮短训练后，Dev仅比同预算普通版多命中1个状态；528状态audit无独立收益，且两版均低于静态基线。详见 [AERIALVG_PAIR_ADAPTATION_RESULTS_20261101.md](AERIALVG_PAIR_ADAPTATION_RESULTS_20261101.md)。该实验与此前自建关系头身份不同，不能合并收益。

## 2026-11-01 原生 AerialVG 分支同步

官方模型理解、原生query训练、因果时序残差、Train实际导航采集及冻结验证证据见 [本次同步索引](reports/aerialvg_20261101/README.md)。新的原生共享评估入口为 `scripts.evaluate_native_shared_closedloop`；现有 `scripts.evaluate_closedloop` 保留历史分辨率与残差评估协议。当前128条试验只有初步正向信号，全量结果尚待核验，不宣称已实现明显提升。
