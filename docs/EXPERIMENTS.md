# 实验协议、性能与复现检查表

**没有真实运行过的结果不得填写数字。**

## 第一阶段：验证架构本身

- CPU tests：shape / backward / signed polygon distance / landmark identity ambiguity / historical RGB observation-only / invisible SigLIP gating
- 1 epoch synthetic smoke，仅查梯度、耗时、显存/内存峰值和 checkpoint
- CityNav teacher JSONL：train_seen，分别汇报 per-epoch loss / global Goal Top1、Top5、Top20 R@20m 与 localization NE
- val_seen、val_unseen 严格各自的真实场景，目标定位不使用 stop oracle

## 第二阶段：图推理消融

| Variant | 说明 |
|---|---|
| G0 | Geometry-only baseline，禁止 RGB |
| G1 | Named landmark binding + 轮廓 |
| G2 | G1 + UAV↔Landmark GATv2 |
| G3 | G2 + Goal↔Landmark proposal GATv2 |
| V1 | G3 + 只利用有效已观测 RGB 的 SigLIP2 |
| T1 | V1 + 多模态人类轨迹监督 |
| S1 | T1 + 校准 Stop（不得破坏 T1） |

关键控制：打乱名称绑定、打乱 land centroid 方向、candidate crop RGB shuffle、随机化描述。上述变化应该引起合理且可测量的 Heatmap/Goal 变化；否则模型可能仅依赖锚点 prior。

## 第三阶段：闭环

需要接入 CityNav 原 env 的 **reset/step/observation**，逐步收集 RGB、位姿、探索足迹、历史，
依据路径第一 waypoint 执行不超过50m的一步，并重规划。报告：
SR / OSR / SPL / NE / nDTW / sDTW，并提供最少 128/128 快速测试和全量 split。

风险检查：
- FOV 外的 RGB 是否被载入？
- candidate crop 的中心是否真实可见？
- GT goal 是否出现在 inference 输入？
- Human future 是否只出现在训练标签？
- Top20 goal recall 是否足够？
- Unseen 的 false-stop 是否可接受？
- 特征缓存是否将不可见变为“黑图负样本”？
- 真实在线轨迹是否避障并且遵循动作预算？

## 训练性能采样

在 2×5090 环境记录每 epoch：samples/s、最大显存、SigLIP2 forward 次数、
平均有效候选数、polygon distance 时间、graph 时间和文本缓存命中率。
现有代码已提供 BF16 与 torchrun/DDP；**不宣称在此环境真实运行过 GPU 训练**。
