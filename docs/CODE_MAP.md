# GraphNav 代码导航与维护边界

本项目保留历史实验以支持复现，**推荐仅从核心模块和正式入口开始阅读**。其余实验脚本请按需查看 [脚本索引](../scripts/README.md)。

## 研究主线

| 功能 | 核心源码 | 入口 |
|:--|:--|:--|
| 正式 GraphNav-V1 C1 | `graphnav/stage1.py`, `graphnav/stage1_runtime.py`, `graphnav/stage1_checkpoint.py` | `scripts/evaluate_closedloop.py --model graphnav-v1`（冻结清单校验） |
| 语言–地标静态定位 A3 | `graphnav/full_localization.py`, `graphnav/candidate_geometry.py` | `scripts/train_a3_replicate.py`, `scripts/train_static_extension.py` |
| A4 GAT 与地标关系图 | `graphnav/semantic_graph.py`, `graphnav/graph.py` | `scripts/train_static_extension.py` |
| 已训练控制器权重迁移 | `graphnav/static_transfer_runtime.py`, `graphnav/a3_runtime.py` | `scripts/evaluate_closedloop.py --model static-transfer` |
| 有界路径、Stop | `graphnav/trajectory_policy.py`, `graphnav/joint_inference_controls.py` | `scripts/train_trajectory_stop.py` |
| 当前/历史视觉证据 | `graphnav/dynamic_belief.py`, `graphnav/dynamic_runtime.py`, `graphnav/observed_landmarks.py` | `scripts/train_dynamic_belief.py` |
| 实验性多尺度图文联合模型 | `graphnav/appearance_matching.py`, `graphnav/joint_active.py`, `graphnav/joint_active_runtime.py` | `scripts/train_joint_active.py` |
| 正式闭环评价 | `graphnav/closedloop.py`, `graphnav/renderer.py` | **`scripts/evaluate_closedloop.py`** |

## 历史与辅助 CLI

- 数据准备：`prepare_citynav.py`、`prepare_static_splits.py`、`prepare_dynamic_evidence.py`、`prepare_joint_active_data.py`、`prepare_trajectory_supervision.py` 等。
- 静态和关系消融：`run_static_controlled_queue.py`、`evaluate_frozen_static.py`、`evaluate_graph_language_factorial.py`、`evaluate_p1_counterfactuals.py` 等。
- 机制诊断：`diagnose_*.py`、`audit_*.py`、`summarize_*.py`、`plot_closedloop_cases.py`。
- 开发阶段合成测试：`python -m graphnav.train --smoke`；不用于报告闭环性能。

这些脚本对应既有报告的输入指纹和历史复现实验。**未经引用关系审计，不要为缩短文件列表而直接删除**：删除可能使已发布指标失去重算入口。后续如需要瘦身，应先将不再依赖的脚本标记为 legacy，并提供等价入口与最小回归测试。

## 命名说明

`graphnav/model.py`、`graphnav/train.py` 是项目早期通用网络/合成 Smoke 入口；**当前实证最优 A3 并不等于 `graphnav/model.py` 的整套早期图架构**。README 的正式复现命令对应 `full_localization.py` 及 `static-transfer`。

## 结果与代码版本

- 对照表与报告放在 `docs/`、`reports/`，不要把训练产物误认成代码依赖。
- 本地权重/缓存统一放 `runs/`（Git 忽略），按校验和与配置关联。
- 进行代码精简、重命名或入口迁移时，必须维护 `tests/`、CLI 兼容和历史数据协议；不允许因为一项实验是负结果就销毁其来源证据。

## main 分支维护状态

- 当前可复现主线是 A3 Geometry + 经来源核验的 Trajectory/Stop，A4 GAT/视觉/History 不自动作为默认。
- GraphNav-V1 正式代码、权重来源清单、Seen/Unseen/Test-Unseen 指标现已归档于 main 的 `reports/graphnav_v1_stage1/`；`.pt` 权重仍需本地核验，不随 Git 分发。
- CI、历史报告路径和配置兼容均是删除旧文件前的检查项。
