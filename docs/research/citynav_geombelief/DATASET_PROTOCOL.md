# 01 · CityNav 数据集和共同评估协议

## 数据版本与分割（已核验的本地文件）

| Split | Original Episodes | Revised Episodes | 移除 | 备注 |
|---|---:|---:|---:|---|
| Train-Seen | 22,002 | 21,878 | 124 | 训练 |
| Val-Seen | 2,498 | **2,470** | 28 | 23 map blocks |
| Val-Unseen | 2,826 | **2,697** | 129 | **4 map blocks** |
| Test-Unseen | 5,311 | **5,281** | 30 | 6 map blocks |

本机发布的 Revised 数据删除总计 311 个 Episode；有些论文报告的移除样本数口径不同，**不能只凭 split 计数确认文件 SHA 完全相同**。保留 Episode 的原始目标坐标、指令和轨迹未变；processed landmark 描述等元数据有变化。详情见 [baseline protocol audit](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/baseline_protocol_audit.md)。

## Query 与 Episode 不混淆

| 用途 | Val-Seen | Val-Unseen | Test-Unseen |
|---|---:|---:|---:|
| 静态/动态目标定位，去重 Query | **2,426** | **2,679** | **5,180** |
| 闭环导航 Episode | **2,470** | **2,697** | **5,281** |

- `R@K`：按每个去重 map+instruction 查询的 20m 半径召回；`R@1` 不等于轨迹最终 SR。
- `NE`：结束位置到 GT 的 **XY 平面欧氏距离** (m)。
- `SR`：最终 XY NE ≤20m，不额外要求成功原因是主动 Stop。
- `OSR`：执行过程中任一采样位置的 XY 距离曾 ≤20m；一般 OSR ≥ SR。
- `SPL`：SR × 起点-目标 XY 直线距离 / max(实际 XY 行进路长, 起点-目标 XY 直线距离)。
- `False-Stop Rate`：失败的 `policy_stop` / 全部 `policy_stop`；没有主动 Stop 的对照组为**未定义**，不得填 0。
- `OSR−SR` 表示到达过却未保留在目标区域的比例，并不直接等于 Stop 模型错误率。
- 每 Episode 最大 20 次宏决策；水平每步 ≤50m，垂直每步 ≤10m；Stop 阈值 0.3，部署时预测目标到达门控 **0.001m**，与评价成功半径 20m 不同。
- 标准目标候选网格 28×28，384×384 是**候选位置采样加密**，不是 RGB 图像分辨率。不同网格和不同训练残差头不可拼接成一个模型。

## 数据和推理权限

- 推理可用：自然语言、公开地图具名地标与几何、当前 UAV pose、当下及**过去**的实际 RGB/历史。
- 推理不可用：GT 目标坐标、未来观测帧、专家剩余轨迹、官方评估标签；Oracle Goal 只允许用作**显式标记的诊断对照**。
- 静态几何表示消融固定三 Seed，使用相同网络容量/训练预算，并审计 geometry/prior/rules 等旁路信息。
- 不同训练配置共享固定旧 Trajectory/Stop 时，闭环变化可能包含**特征分布迁移**；原完整模型闭环必须与迁移实验分开。

## 审计与证据

- [数据/基线版本核对](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/baseline_protocol_audit.md)
- [正式完整模型结果](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/graphnav_v1_stage1/GRAPHNAV_V1_RESULTS.md)
- [实验资产索引](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/experiment_registry.csv)
- [地图几何信息通路审计](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/polygon_ablation/POLYGON_FEATURE_AUDIT.md)

**论文注意：** 外部方法的原论文值、与我们相同 XY evaluator 的本地重算、原始数据版本和替代 API 模型，必须分别标明。不能把“共用 Revised 分割”直接写成严格同训练/同传感器/同控制器复现。
