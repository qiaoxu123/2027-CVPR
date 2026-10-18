# 02 · CityNav / GeomBelief 已完成实验与证据台账

> 截至 2026-10-18。**正式归档模型 C、独立视觉残差、384 网格、重新训练的几何模型**是不同实验身份。以下表格保留负结果和不同权重限制。凡统计区间未做多重比较校正，均不得宣称普遍显著性。

## E01 正式完整模型：闭环导航（三 Split）

原归档模型 SHA-256：`00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`；28×28；Stop 0.3。

| Split | Episodes | SR % | OSR % | SPL % | NE m |
|---|---:|---:|---:|---:|---:|
| Val-Seen | 2470 | **47.6113** | 55.6680 | 46.0028 | 29.1192 |
| Val-Unseen | 2697 | **22.5436** | 31.7019 | 21.8205 | 51.5327 |
| Test-Unseen | 5281 | **37.1521** | 46.5632 | 35.8061 | 35.0079 |

同一权重的初始→动态最终静态定位 R@1：Val-Seen 47.197→47.444；Val-Unseen 22.023→22.471；Test-Unseen 36.815→37.201。**说明原视觉/历史修正的定位排序变化有限，不能单凭静态热图外观证明不存在修正。** [证据](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/graphnav_v1_stage1/GRAPHNAV_V1_RESULTS.md)。

## E02 候选网格分辨率（同冻结权重；全量三 Split）

| Grid | Seen SR/SPL % | Unseen SR/SPL % | Test-Unseen SR/SPL % |
|---|---|---|---|
| 28×28 | 47.61 / 46.00 | 22.54 / 21.82 | 37.15 / 35.81 |
| 384×384 | 48.83 / 46.95 | 23.17 / 22.32 | 38.27 / 36.67 |

候选数 784→147,456，约 188×；性能小幅改善但算量明显增大。**384 的收益不可加到独立校准残差上。** [汇总](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md)。

## E03 三 Seed 几何表示：Centroid / Box / Full Polygon（完成）

15 次独立训练、三 Seed、54 次实际全分割闭环评估；相同 10-D 容量和旧固定控制器。Val-Unseen 均值±标准差（%）：

| Representation | R@1 | SR | SPL |
|---|---:|---:|---:|
| Centroid | 19.30±1.28 | 19.49±1.18 | 18.88±1.15 |
| Bounding Box | 19.27±1.97 | 19.28±2.16 | 18.71±2.14 |
| Full Polygon | **20.00±1.31** | **19.75±1.26** | **19.18±1.27** |

Full−Centroid Unseen ΔSR +0.26pp，map-cluster CI [−2.53,+3.06]；Full−Box +0.47pp，CI [−0.47,+1.85]。**不能宣称完整多边形稳定优于中心点/矩形。** 现有权重去掉边界距离后 Seen/Unseen R@1 分别下降 5.69/5.38pp，属于**冻结网络特征敏感性**，不是轮廓复杂性因果证明。[原报告](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/polygon_ablation/POLYGON_ABLATION_REPORT.md)。

## E04 地图几何独立贡献（核心实验 Done）

三个匹配 Seed，完整静态定位及闭环 Seen/Unseen。Val-Unseen 均值（%）：

| Language + input | R@1 | SR | SPL |
|---|---:|---:|---:|
| Name/Type Only | 0.61 | 0.66 | 0.62 |
| Shuffled Geometry | 3.66 | 3.65 | 3.54 |
| Direction Only | 6.47 | 6.35 | 6.16 |
| Distance + Direction | 18.64 | 18.71 | 18.13 |
| Distance Only | 19.70 | 19.75 | 19.15 |
| **Correct Geometry** | **20.00** | **19.75** | **19.18** |

- 正确几何−Name/Type：Unseen R@1 +19.385pp、CI [+11.407,+26.104]；**SR +19.095pp、CI [+11.047,+25.955]**。
- 正确几何−Shuffle：Unseen R@1 +16.337pp、CI [+8.299,+22.869]；**SR +16.104pp、CI [+8.009,+22.662]**。
- Correct−Distance Only：Unseen R@1 +0.299pp，CI [−0.888,+2.169]；**SR 差为 0pp**，CI [−1.188,+1.938]，尚未证明复杂几何优于简化距离信息。
- 打乱的是成组空间几何上下文（同时含 prior/rules 等），不是纯地标名称绑定位置信息；“Name/Type Only”仍存在候选绝对位置网格，可学到无条件位置先验。**不能仅凭此证明唯一贡献是语言—几何交互。**
- 三 Seed 的定位器接旧固定控制器；此闭环指标与 E01 非同一权重的完整模型不可按简单差值推断收益。

[独立几何报告](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/MAP_GEOMETRY_ABLATION_REPORT.md) · [总结与置信区间](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/GEOMBELIEF_CORE_CONCLUSIONS.md) · [验收记录](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/acceptance.json)。

## E05 校准视觉残差（独立重训 Head；全 Seen/Unseen）

| Val-Unseen 对照 | SR % | SPL % | OSR % | NE m |
|---|---:|---:|---:|---:|
| No-belief | 22.14 | 21.53 | 31.29 | 51.76 |
| Zero | 22.14 | 21.53 | 31.29 | 51.76 |
| Shuffle | 22.62 | 20.81 | 34.74 | 51.63 |
| **Real RGB** | **23.95** | **22.76** | 34.56 | **50.68** |

Seen Real SR 48.70%，No-belief 47.37%；Unseen Real−No-belief ΔSR **+1.8168pp**，整地图 95% CI **[0,3.441]**；ΔSPL +1.2329pp，CI [−0.656,2.796]。Zero/No-belief 逐条等价，Shuffle 在 SPL 上较低；**不能据此宣称对新城市具有稳定提升**（仅四个独立 Unseen 地图块）。[原报告](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md)。

## E06 冻结残差输入干预（已测，不是公平重训）

Calibrated Residual 上固定所有网络，屏蔽不同特征，Val-Unseen：

| 视觉输入方案 | SR % |
|---|---:|
| Geometry + Pose Only（冻结输入屏蔽） | 22.62 |
| RGB + Text Only（冻结输入屏蔽） | 24.73 |
| Current RGB Only（禁用历史） | 22.88 |
| Real（原设置） | 23.95 |

Real−Geometry/Pose SR +1.335pp，地图 CI [+0.578,+2.010]；Real−RGB/Text **−0.779pp**，CI [−2.486,+0.604]。这些对照**证明冻结头对输入敏感**，不能当作重新训练了 Geometry-only/RGB-only 的公平方法比较。下一步公平重训尚 Pending。[原报告](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/GEOMBELIEF_CORE_CONCLUSIONS.md)。

## E07 轨迹/Stop 控制器因果诊断（全部 Done）

现有正式 C 权重不变；一组 Native Dynamic Belief 让路径改变后新观测随之变化，另一组 Identical Initial Belief 在所有策略中共享相同首帧候选场，区分执行与观察耦合影响。

| Native Dynamic (Val-Unseen, 2697) | SR % | SPL % | OSR % |
|---|---:|---:|---:|
| Direct-to-Goal | **22.69** | **22.05** | 30.63 |
| Guarded-Blend | 22.58 | 21.85 | 31.70 |
| Raw Learned Waypoint | 20.17 | 14.33 | 40.53 |
| Rule Arrival (same 0.001 m gate) | 22.58 | 21.85 | 31.70 |
| Oracle Goal + Direct（仅诊断） | 100.00 | 100.00 | 100.00 |

- 在 Identical Initial Belief 下，Direct 与 Guarded-Blend Unseen SR 相同（22.32%）；Guarded-Blend 的 SPL 略低。**尚无学习轨迹改进目标追踪 SR/SPL 的独立证据**。
- Raw Learned Waypoint 在动态条件下的 Unseen SPL 比 Direct 低 **7.723pp**（地图区间不跨零）；说明未经保守约束的开放轨迹建议可能导致绕行/错过最终目标。
- Learned Stop 与规则 Arrival（相同 0.001m 门控）的 SR 基本相同；**不能声称 Stop Head 独立创造了导航收益**。
- Oracle 替换 GT 到控制器仅是评估上限，不可部署，也不可将 +77pp 直接当成“定位错误独占比例”。
- Native 新运行时与历史正式 Rollout 有微小数值漂移；**E01 正式结果不能被新运行时结果覆盖**。[完整日志及两套控制口径](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/geombelief_remaining/CONTROL_ATTRIBUTION_REPORT.md)。

## E08 Stop / Arrival Guard 与失败诊断

原模型 Val-Seen 完整 2470：到达门控阈值 0.001/1/5/10/20m 对应 SR 47.61/47.57/47.37/47.00/45.10%；取消门控 SR 43.97%。这是规则 Guard 的敏感性，**不是 Learned Stop 的单独优势证明**。正式模型的 Val-Unseen False-Stop/Stop 约 77.44%，需要核对预测目标误差与终止之间的关系。[正式结果](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/graphnav_v1_stage1/GRAPHNAV_V1_RESULTS.md)。

## E09 Profiling / E10 公开参考 / E11 真实案例

- 全量 Unseen Profiling、HETT 历史耗时、冷启动详见 [04 效率台账](EFFICIENCY_BASELINES.md)。
- 外部 MGP、HTNav、HETT、FlightGPT、GeoNav 原论文或本地替代模型应按 Original/Revised、评估器与模型来源分表；GeoNav (DeepSeek Flash) 使用 Original Test 5311，不与当前 Revised Test 5281 直接排名。详见 [baseline audit](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/baseline_protocol_audit.md)。
- 已有语义/可见性与信念热图、动态初末差分、错误终止案例存于工程历史报告。真实图像与 Episode ID 要随新定性图归档，不以示意图作为“实测证据”。

## E12 历史 HETT / SBF 复现实验（内部资料）

历史研究含 HETT 粗定位、两阶段探索、SBF 系列续训以及历史 128-Episode 复现实验。这些记录用于解释演化和定位失败机制；**没有正式同条件对齐的模型和指标，不作为本文正式公开 Baseline 排名**。勿混淆新模型与历史权重。详见 [基线协议审计](https://github.com/qiaoxu123/2027-CVPR/blob/research/visual-residual-20261013/reports/paper_experiments/baseline_protocol_audit.md)及项目历史代码/报告。
