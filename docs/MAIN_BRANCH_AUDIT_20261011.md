# main 分支配置核查与保守精简（2026-10-11）

## 范围

检查远端 `qiaoxu123/2027-CVPR-Dev`，基线 `main@0c2584421f812821791091f734371380b759c24d`。本次属于**工程审计和不改变数学计算的维护**，不是重新训练，也不声称已经重新验证全量 GPU 指标。

## 当前应当采用的发布主线

- **默认静态定位骨干：A3 Geometry**，不是 A4 GAT。见 `graphnav/full_localization.py`，实际调用时明确指定 `variant="geometry"`。
- **控制器：**已训练、来源兼容的 Trajectory/Stop + guarded-blend + 公开高度范围校验。需要从 `graphnav/static_transfer_runtime.py` 的来源约束进入，避免意外随机重置控制器。
- **推理入口：**`scripts/evaluate_closedloop.py --model static-transfer`；128-episode 开发评估与全量 Seen/Unseen 严格分列。
- **实验模块：**A4 GAT、RGB/History、RegionCLIP/RemoteCLIP 与双向回程任务保留供进一步研究，不因模块可运行就打开为默认导航配置。

用户提供的已完成本地全量结果：

| 本地方案 | Seen SR/SPL | Unseen SR/SPL |
|---|---|---|
| A3 强基线 | 48.50% / 47.02% | 21.25% / 20.64% |
| GraphNav-V1 C1 | 47.61% / 46.00% | 22.54% / 21.82% |

这些数值尚未通过当前远端 `main` 的独立权重与报告重放核验；GraphNav-V1 本地提交 `9e458d7` 在远端不可获取，`runs/graphnav_v1_stage1/best.pt` 不在 Git（按设计忽略）。因此**无法宣称 main 已包含本地 V1 最优权重或已复现其 Unseen 优势**。后续需仅在用户将本地来源/清单同步后决定正式默认模型。

## 已发现并修复的工程问题

1. 最新 main 的 GitHub Actions `cpu-unit-tests` 在 `0c25844` 失败：`tests/` 中7个模块因 `ModuleNotFoundError: experiments` 而中止收集，CPU 训练 Smoke 没有运行。这是 Python 包安装/测试路径问题，非训练指标失败。
2. `pyproject.toml` 原 `[tool.setuptools.packages.find]` 仅包含 `graphnav*` 与 `scripts*`，未涵盖已使用的 `experiments*`。增加后统一采用 `python -m pytest -q`，并对实验目录进行 `compileall`。
3. 根 README 引用了不存在的 `scripts/README.md`；本次补齐常用入口与研究实验边界。
4. 删除核心模型内**经静态检查确认未使用的 import**，且把地标几何归一化的长行拆开，提高可读性，不改变公式、参数名称或张量形状。

## 未擅自进行的操作

- 没有把 `main` 的 `forward(variant="gat")` 历史默认参数直接改成 `geometry`；这样会改变旧脚本和隐式调用者语义，必须先审计调用处再迁移。
- 没有改写 A3 权重、模型结构、大小、候选评分或视觉 Gain；没有对 `main` 的 Unseen 进行重新调参。
- 没有将 `research/*` 分支合并成主模型：10月11日视觉身份与多视角报告未证明可靠的 Goal 恢复；多视角点云渲染因观测覆盖/身份验证门槛不足而停止扩展。
- **没有盲删历史 reports/scripts/tests：** Git Blob SHA 审计发现少量完全重复的归档 JSON/CSV，但它们位于不同实验的固定复现路径；现阶段无法证明无外部报告或重放流程引用。重复不等于无用，先保留。未来可通过全仓引用扫描、实际入口重放和归档重定向后再删除。
- 没有把视觉论文指标/弱标签当成人工核验语义成绩，也没有将新的探索性模块默认启用。

## 单卡 RTX PRO 6000 96GB

正式训练和 GPU 评估应在一张 RTX PRO 6000 96GB 上调度；CPU 数据审计可并行，避免文档再建议两卡 5090。

## 验证要求

- `python -m compileall -q graphnav scripts experiments tests`
- `python -m pytest -q`
- `python -m graphnav.train --smoke ...`（纯合成 CPU Smoke）
- 在本地存在数据和权重时，用同一 A3 权重与原控制器作小样本逐 Episode 完全一致性回放，**再**开展正式 Seen/Unseen。GitHub 线上 CI 不应访问私人 CityNav 原始数据或未上传的 Checkpoint。

以上所有**实际执行**的步骤应由 GitHub Actions 状态及工作记录证明；未运行的本地 GPU 重放不能写成已完成。
