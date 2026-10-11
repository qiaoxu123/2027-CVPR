# GraphNav 脚本入口（main）

日常使用优先从本文件列出的入口开始。其余 `audit_*`、`diagnose_*`、`run_*_queue` 等脚本属于历史实验或专项诊断；不是默认模型配置。

## 推荐主线

| 任务 | 入口 | 说明 |
|---|---|---|
| **GraphNav-V1 正式评估** | `scripts/evaluate_closedloop.py --model graphnav-v1` | C Epoch1，Unseen 必须指定已冻结的候选清单和有效权重 SHA |
| **GraphNav-V1 恢复/汇总** | `scripts/run_graphnav_stage1.py`, `scripts/summarize_graphnav_v1_results.py` | 保留训练结果和来源完整性 |
| 预处理 CityNav | `scripts/prepare_citynav.py` | 需要本地已授权数据 |
| 静态 A3 训练/复现 | `scripts/train_static_extension.py` | 必须明确指定 geometry 变体与匹配配置，不假定 GAT 最优 |
| A3 / A4 定位评估 | `scripts/evaluate_frozen_static.py` | 静态 R@K、NE；不等于闭环导航 |
| 正式闭环评估 | `scripts/evaluate_closedloop.py` | 单一推荐闭环入口；具体参数见根 README 与复现协议 |
| 轨迹/Stop 训练 | `scripts/train_trajectory_stop.py` | 核验来源 Checkpoint/配置，不能随机初始化后冒充复现 |
| 视觉/History 研究 | `scripts/train_joint_active.py` | 实验性；不作为已证明优于 A3 的默认模型 |

## 实验归档

- `experiments/visual_grounding/`：RemoteCLIP / RegionCLIP / SigLIP2 对照及负结果。
- `experiments/bidirectional_core/`：正向/反向轨迹和视觉弱监督小规模验证。
- `scripts/run_static_controlled_queue.py`、`scripts/run_a4_batch_followups.py` 等：历史受控消融入口，保留以复现已有指标。

## 复现注意事项

- 先执行 `python -m pytest -q`，正式 GPU 运行前再做真实数据/权重 CUDA Smoke。
- 正式完整导航模型为 GraphNav-V1 C Epoch1；独立 A3 Geometry 保留强基线。V1 包含 RGB/History，但尚未证明它们有稳定的独立增益。
- 新的 V1 最佳权重若仅保存在本地，不能仅依赖远端仓库重新产生相同结果；请以权重 SHA 和对应报告核验。
- 禁止用测试集挑选模型；请勿覆盖 `runs/` 的原实验 Checkpoint。
- 清理代码时不得删除报告中引用的历史脚本或来源审计文件。参见 `docs/CODE_MAP.md` 与 `docs/REPRODUCIBILITY.md`。
