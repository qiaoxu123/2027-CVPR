# 03 · 研究假设、验证队列与可否证门槛

> 状态：**Supported / Not supported / Mixed / Pending / Historical**。结果不是“模型越复杂越好”的筛选依据；保留失败假设。任何新增工作先写预注册方案和判据，结束后再更新结果。

## 当前假设台账

| ID | 可检验假设 | 结果与解释 | 状态 | 证据 |
|---|---|---|---|---|
| H01 | 正确地标几何显著优于只有名称/类型 | Unseen SR +19.095pp；4 map-block CI >0；模型仍能使用绝对候选网格先验 | **Supported（限定条件）** | [几何核心结论](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/GEOMBELIEF_CORE_CONCLUSIONS.md) |
| H02 | Full Polygon 可稳定优于 Box/Point | +0.47/+0.26pp SR，但地图 CI 跨零；不足以证明必要性 | **Not supported** | [Polygon报告](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/polygon_ablation/POLYGON_ABLATION_REPORT.md) |
| H03 | 全几何比 Distance Only 明显更强 | Unseen SR 完全相同，R@1 +0.299pp CI 跨零 | **Not supported / mixed** | [地图几何报告](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/MAP_GEOMETRY_ABLATION_REPORT.md) |
| H04 | 正确 RGB 能稳定提高新地图 SR | Real−No-belief Unseen +1.817pp，地图 CI 下界 0；vs Shuffle 更有力 | **Mixed** | [视觉对照](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md) |
| H05 | RGB 作用不只是几何/位姿上下文 | Frozen Real−geometry_pose +1.335pp CI >0；尚无公平重训因果结论 | **Supported as input sensitivity** | [视觉输入干预](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/visual_input_attribution.csv) |
| H06 | Learned Waypoint 比直接朝目标更好 | 动态 Unseen SPL 下降 7.723pp；Guarded Blend≈Direct | **Not supported** | [控制器报告](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/CONTROL_ATTRIBUTION_REPORT.md) |
| H07 | Learned Stop 优于相同到达规则 | SR 等同；强 0.001m 门控主导现有执行逻辑 | **Not supported** | [控制器报告](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/CONTROL_ATTRIBUTION_REPORT.md) |
| H08 | 飞行过程的信念更新显著改变目标排序 | 原模型 Unseen 初始 R@1 22.02→动态最终 22.47；增量较小 | **Limited** | [正式结果](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/graphnav_v1_stage1/GRAPHNAV_V1_RESULTS.md) |
| H09 | 当前系统严格快于 HETT | 历史表比值 Rollout 4.17×，但未控制测量环境、模块边界与并发 | **Pending matched benchmark** | [效率归档](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/efficiency_existing_evidence.json) |

## 未来最值得验证（优先级）

| 优先级 | 新假设 / 问题 | 最小设计 | 判定门槛 / 避免混杂 | 现状 |
|---|---|---|---|---|
| P0 | **Language×Geometry 交互是否真正必要？** | Correct geometry + matched instruction、同地图内语言指令 Shuffle、正确 Geometry + 几何无条件 spatial prior、可选无几何 point-regression 输出 | 同容量/相同训练预算/三个 Seed；Unseen 地图配对 R@1/SR CI，检查语言匹配是否超出位置先验 | Pending |
| P0 | **哪些 Goal 错误不可由控制器修复？** | 从已完成 Controller 两套视角中离线分层：定位命中、移动错误、访问但离开、错误 Stop | 固定原预测/轨迹文件；按地图统计、识别同一 Episode 的失败转移；不重复全量 Rollout | Analysis pending |
| P1 | **视觉 RGB/文本/几何证据的公平因果贡献？** | Geometry+Pose / RGB+Text / Current RGB 三个 residual 头重新训练与固定原始设置对照 | 同网络预算与训练 Seed，先单 Seed 审计后决定扩展；只用 Train-Dev 选模型；Val-Unseen 仅评估 | Fair retraining Pending |
| P1 | **跨地图置信区间是否稳定？** | 新独立地图或跨城市数据，预先冻结模型与参数 | 不能把四个 Unseen map-block 当四个城市；CI 含地图簇分层及 3 Seed | Pending |
| P1 | **候选分辨率值得牺牲计算效率吗？** | 28/56/112/384 逐级候选数与耗时测量（若确需） | 相同 checkpoint、相同测量环境，报告 SR/SPL/Latency/Memory Pareto | Partial |
| P1 | **视觉修正改变了什么位置？** | 利用已有 logits、mask、Top1、Belief Δ 和真实航迹离线统计 | 先验证每步缓存存在，拒绝想象合成轨迹；分析可见区域内/外更新 | Partial |
| P2 | **HETT 与当前系统能否同机公平测速？** | 同硬件/同渲染协议/同冷启动规范/无并发环境 profiling | GPU Event 与 host wall-time 边界相同；报告 memory、model-only、rollout | Pending |

## 新假设的统一记录模板（复制一个条目）

### Hxx · 问题标题
- **提出时间 / 提出人：** YYYY-MM-DD / —
- **假设（可否证）：** “若因素 A 是必要因素，则控制 B 后 X 指标应有方向明确的变化”。
- **已有证据与反例：** 报告链接、原始数量、不同模型或数据版本等限制。
- **对照组：** 控制变量、实验组、匹配种子、网络容量和输入权限。
- **指标与分母：** SR/SPL/NE/OSR 或 R@K；Split、Query/Episode、map-block 数。
- **统计计划：** 优先 whole-map cluster bootstrap；Seed 不是独立地图；多重比较单独解释。
- **通过门槛：** 主要指标效应方向、阈值、区间与预算；同时定义“失败时怎样修订论文主张”。
- **执行状态：** Proposed → Ready → Running → Done / Blocked。
- **产物与验证：** Config/Checkpoint SHA/Git Commit/逐样本预测/实验日志/统计脚本/图表。
- **结论：** Supported / Not supported / Mixed，列清不可归因部分。
- **后续：** 直接链接下一假设，不重跑已完成基线。

## 更新规则

1. 仓库 `EXPERIMENT_LEDGER.md` 保留实验真值与指向不可变 Checkpoint 的路径；此文件保留假设的来龙去脉与可否证条件。
2. Outline 维护同一内容的协作入口，新增讨论可以用评论，验证结论必须更新 Markdown 正文并带证据。
3. 每次同步附上源 Commit / 日期；记录历史结果为 Historical，而非替换原分析。
4. 禁止把 Frozen Input Intervention 标为同预算重训；禁止把 Oracle Goal 作为可用导航方法。
