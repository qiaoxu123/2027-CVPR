# GraphNav 复现与评估协议

此文档记录**已实现入口**与**已验证实验**，供代码发布及后续模型训练使用。不能将实验计划当成已完成结果。

## 1. 环境与数据依赖

- Python 3.10/3.11；PyTorch 需与 CUDA/显卡适配。
- 安装：`python -m pip install -e '.[dev,data,foundation]'`，并安装 `pyyaml`。
- 本地需要 Revised/processed CityNav、城市地图地标、授权 RGBD 数据，以及本地 SBFNav 复现源码。私有路径、CityNav 原始图像和模型大文件均**不在仓库**。
- 所有正式评估先运行 `pytest -q` 和 CPU Smoke；GPU/外部数据环境必须另做 CUDA Smoke。
- 验证数据集不能用于拟合参数。Val Unseen 不得用于调超参数或选择最优训练轮次；最终 Test Unseen 应遵守对应冻结协议。

## 2. Checkpoint 来源（关键）

| 模型 | SHA-256 | 使用说明 |
|:--|:--|:--|
| A3 Seed1 Epoch14 | `9d9be861bf7e9cac986d747867990de3d409ccb9bc1545862d71a620448380d7` | 新静态强候选 |
| 原 Trajectory/Stop | `51d8af7f05a9a718a9dc0ed12721a24ca6dd252e2714cb25948f22cb88cd2dbd` | 已训练，严格兼容迁移 |
| 控制器原参考 A3 | `2008085175d3da5f84fa1fc880d3ef38ba3c3fdd7595bd33a3c856f22d6baf2c` | `static-transfer` 用于来源核验 |

上述仅是 **SHA 标识符**，不是仓库提供的可下载权重。历史实验本地位置和完整 provenance 见 `reports/a3_relation_gat_20261010/final/summary.json`。迁移应通过 `graphnav/static_transfer_runtime.py` 兼容性验证，不能在新静态模型下偷偷随机化 Stop/路径头。

## 3. 已验证指标及分母

- 静态：Val Seen 去重指令 2,426 Query；R@K/20m 表示前 K 候选内有至少一个距离真实 XY ≤20m。
- 新静态 A3 Epoch14：R@1 48.4749%，R@4 68.6315%，R@8 78.0297%。
- 新强基线闭环：固定 Val Seen 128 Episodes，SR48.4375%，SPL47.4130%，OSR56.25%；已核对原训练控制器的迁移。
- 旧 A3 全量 Seen：2,470 Episodes，SR40.89%、SPL39.77%。**与新模型 Seen128 不同。**
- 本地 SBFNav Epoch14：全量 Val Seen 2,470 Episodes，SR43.5223%、SPL36.1184%；Val Unseen 2,697 Episodes，SR21.5054%、SPL15.8907%；来自本地独立复现而非官方原仓库权重。

每个结果必须输出：原始/修订 CityNav 版本、分母、候选去重协议、模型选择 split、2D/3D 度量、Stop 条件、Checkpoint SHA、代码 Commit、配置及逐 Episode 指标。Seen128 与全量值永远分开展示。

## 4. 单一强基线评估

设置本地环境变量，不要直接复用报告中原开发机的绝对路径：

```bash
export SBF_SOURCE=/path/to/SBFNav-CityNav-Reproduction/source
export SBF_CONFIG=/path/to/SBFNav-CityNav-Reproduction/configs/sbfnav/main.yaml
export CITYNAV_DATA=/path/to/citynav-data
export A3_BEST=/path/to/E1_seed1/best.pt
export REFERENCE_A3=/path/to/reference_geometry_best.pt
export HEAD_BEST=/path/to/a3_trajectory_stop_best.pt

python scripts/evaluate_closedloop.py \
  --model static-transfer --sbf-source "$SBF_SOURCE" \
  --config "$SBF_CONFIG" --data-root "$CITYNAV_DATA" \
  --checkpoint "$A3_BEST" \
  --reference-a3-checkpoint "$REFERENCE_A3" \
  --head-checkpoint "$HEAD_BEST" \
  --stop-threshold 0.3 --head-execution guarded-blend \
  --split val_seen --max-episodes 128 \
  --output runs/graphnav_release_seen128

# 全量 Seen：移除 --max-episodes；使用新的输出目录
```

`scripts/evaluate_closedloop.py` 会拒绝覆盖已有输出目录。正式评估采用共享宏动作控制：最多 20 步、单步水平上限 50m、垂直 10m。不要将“达到目标附近后仍飞走的 OSR”误算成 SR。

## 5. 联合训练尚不能直接视为完成

现有 `scripts/run_joint_active_20e.py` 和 `scripts/train_joint_active.py` 是**实验入口**。它们创建新的 `JointActiveGraphNav` 和 `CandidateTrajectoryStop`；默认并不从旧已训练 Trajectory/Stop 权重执行兼容迁移。因此未经修改和验证，不应把强 A3 作为输入后直接跑 20 Epoch 并期待保留旧导航性能。

历史旧联合模型 Epoch1（启动主动观察）Seen128 SR28.91%，不是推荐发布模型。相同权重在被动、预测目标 Stop 门控下需要独立配对评估。详见 `docs/JOINT_EPOCH1_CONTROL_FIX_AND_ABLATIONS.md`。

建议第一阶段交付流程：静态强基线的 Seen128/FullSeen → 冻结 A3 和原控制器训练可见 RGB / History → 严格零视觉/交换视觉消融 → 保持已经训练的所有权重进行低学习率联合微调 → 冻结 Seen 选出的最终候选 → 全量 Unseen。

## 6. 审计与结果保存

- `reports/a3_relation_gat_20261010/`：E1/E2/E3、Seed、GAT 和配对 CI。
- `reports/sbf_protocol/`：历史共享分母、控制器和输入来源审核。
- `reports/active_perception_20261010/`：视觉质量、主动观察与旧联合训练诊断。
- `docs/GRAPHNAV_SBF_INPUT_PROCESSING_AUDIT_20261010.md`：训练/推理模态差异及防信息泄漏说明。

正式权重保存在受控本地 `runs/` 等目录，**不要把权重、原始数据、个人凭据提交 Git**。报告 JSON、CSV、少量审计样例属于可复核的历史证据；不代表它们都应该成为推理时的依赖。

## 7. 已知局限与数据使用

当前 RGBD 观测是 CityNav 正射 RGB + 垂直深度代理，不是真实透视相机的三维遮挡/碰撞仿真。所有 RGB 特征必须来自该 Episode 当前及历史可见视场；GT 坐标与人类未来轨迹只能用于监督或评估，不能进入 `forward()`。视觉匹配依赖当前公开地图和图像配准，但不能把几何 ROI 当完美实例分割。
