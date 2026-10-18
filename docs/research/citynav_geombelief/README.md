# CityNav / GeomBelief — 实验知识库与假设验证

> 最近整理：2026-10-18 · 数据来源：`2027-CVPR` 已提交报告与 Outline 历史笔记。**本目录是带证据边界的研究台账，不替代论文主表或原始 CSV/JSON**。
> 工作名称 GeomBelief；正式完整模型原始权重仍沿用 GraphNav-V1 C Epoch1 的归档身份。禁止通过改名、拼接分割或合并独立消融宣称一个不存在的新模型。

## 导航

| 文件 | 维护目的 |
|---|---|
| [01 数据集与评估口径](DATASET_PROTOCOL.md) | split/Query/地图簇、指标、输入权限、协议差异 |
| [02 已完成测试与结论](EXPERIMENT_LEDGER.md) | 每项独立配置、指标、对照、证据与不支持的推断 |
| [03 假设与验证队列](HYPOTHESIS_REGISTER.md) | 已验证/不支持/受限/待测试；新增实验模板与门槛 |
| [04 Profiling 与外部参考](EFFICIENCY_BASELINES.md) | 成本结构、冷启动、历史 HETT 耗时与基线可比性 |

## 核心结论（带限定）

1. **地图几何有效：** 三 Seed、正确几何 vs. Name/Type Only 的 Val-Unseen R@1 **+19.385 pp**，闭环 SR **+19.095 pp**，四地图块区间均在零以上；相较打乱地图几何亦有正向收益。但不能排除无条件地图位置先验等其他空间线索，不能仅凭此证明语言—几何交互是唯一来源。
2. **完整轮廓并非关键：** Polygon vs. Centroid/Box 的 Val-Unseen SR **+0.26/+0.47 pp**，地图级 CI 均跨零；距离类简单特征几乎达到正确完整几何的平均效果。
3. **视觉修正有一定平均增量：** 独立校准视觉残差 Unseen Real 相比 No-belief SR **+1.8168 pp**，地图级 CI **[0, 3.441]**，不能宣称稳定的跨地图收益；Zero 与 No-belief 相同，Shuffle 削弱 SPL。
4. **控制策略不必更复杂：** 正式冻结网络下 Direct-to-Goal 与 Guarded-Blend 的 SR 接近；原始 learned waypoint 显著降低 SPL，learned Stop 相对相同到达规则没有可分辨增量。
5. **Profiling 已完成：** 2,697 Val-Unseen Episodes、22,144 决策样本，Rollout 均值 48.17ms，模型 GPU 前向 24.33ms，视觉编码 13.13ms，峰值 allocated 3.78GiB。与 HETT 的历史耗时表不得冒充同硬件、同计时边界加速比。

## 记录规范

- 一行测试 = 一个**具体模型权重 + 数据版本 + split + 指标定义 + 控制器 + 输入消融 + 随机种子**，不按方法名字去重。
- `Done` 仅代表对应评估实际完成，`Pending` 绝不填预测数据。`Reused` 指历史数据引用，不等于新训练。
- 仅 Train/Dev 允许调参与选模型；Val-Unseen/Test 结果用于评估而不是回头挑选配置。
- 对每条结论附上报告路径、Checkpoint SHA/Commit、置信区间与无法排除的解释。四个 Val-Unseen map blocks 不是四座独立城市。
- 新假设先登记 **预期可否证条件**，再训练；失败结果同样保留。
- 原始研究历史可保留在代码与报告中；正式论文比较使用有可比协议的数据。
- Outline 位置：成员空间 / 乔旭 / 3.论文写作与工程 / 2027-CVPR / 实验知识库。
