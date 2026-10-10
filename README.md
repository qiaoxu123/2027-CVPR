# GraphNav — Graph-First Aerial Vision-and-Language Navigation

> **当前主实验：GraphNav-V1，第一阶段训练与正式评估已完成。** 冻结的 C Epoch1：Val-Seen 2470，SR47.61% / SPL46.00%；Val-Unseen 2697，SR22.54% / SPL21.82%；Test-Unseen 5281，SR37.15% / SPL35.81%。主结果只列 GraphNav-V1；历史架构说明与对照保留归档。完整指标、执行协议和限制见 [GraphNav-V1 正式结果](reports/graphnav_v1_stage1/GRAPHNAV_V1_RESULTS.md)。本页下方早期“尚无闭环成绩”等描述属于历史研发状态。

> **GraphNav P0–P2 research**: [Bidirectional geometry + causal landmark RGB pretraining](docs/BIDIRECTIONAL_VISUAL_P0_P2.md). Based on validated A3/A4/A5 results; retains A3 baseline, adds world-polygon→actual RGB ROI auditing, valid inverse relation pairs, SigLIP2 Large patch-token cross-view landmark contrastive pretraining. Does NOT claim closed-loop SR/SPL improvement.

> **NEW offline full-token + RGB A0–A5 experiment:** [FULL_TOKEN_GEOMETRY_RGB.md](docs/FULL_TOKEN_GEOMETRY_RGB.md) — tokenwise BERT cross-attention, per-candidate true polygon geometry, typed GAT and *causal observed-only* SigLIP2 evidence. Explicit Original/Revised CityNav protocol and audited fuzzy matching; RTX Pro 6000 96GB, batch 16. **Not yet a closed-loop navigation controller.**
> **Extended CityRefer grounding experimental branch:** Train static goal localization with ALL typed named surface polygons (TrafficRoad/Footpath/Bridge/Parking/Ground), conservative aliases and a buildings-only control. See [FULL_SURFACE_GROUNDING.md](docs/FULL_SURFACE_GROUNDING.md). Rerun data conversion with `--refresh-raw`; historical metrics do not automatically carry over.

> **Static-first research branch**: for Prior → Geometry Rules → Typed GAT localization ablations on a single RTX Pro 6000 96GB (batch 16), see [static-geometry protocol](docs/STATIC_GEOMETRY_ABLATIONS.md). RGB/Trajectory/Stop are deliberately excluded until static goal localization gains are established.


**全新独立架构。** 针对 CityNav / CityRefer 的图驱动无人机语言导航研究工程。
不使用 HETT 的 Transformer、Heatmap Head、Two-Stage Controller、预训练权重或其模型文件。

> **当前状态（研究基线）**：包含图网络、稠密目标概率场、可见性约束 RGB 记忆、冻结 BERT/SigLIP2 接口、候选图排序、多模态路径头、CityNav 数据转换、CPU 测试及训练入口。尚未完成 CityNav 在线仿真器的动作/环境适配，因此 **尚无真实闭环 SR/SPL 数值**，不应将开放式训练损失当作导航性能。

## 设计

\`\`\`text
                          Static at episode reset
 CityNav landmarks (name, ID, ordered contour)  Instruction
                      |                         |
                  deterministic             frozen BERT
                 name binding                   |
                      v                         v
               Contour Encoder -----------> Named Landmark Nodes
                      |                         |
          UAV pose -->|------------> Edge-Conditioned GATv2 (2 layers)
                      |                         |
                      +--------> Dense Goal Field over 28x28 XY grid
                                        |
                              Top-20 goal hypotheses
                                        |
                            Goal / Landmark / UAV GATv2
                                        |
 Observed-only RGB memory -- SigLIP2 ----+-- Visibility-gated evidence
                                        |
                        joint goal + trajectory mode ranking
                                        |
                      3 candidate paths x 8 waypoints / goal
                                        |
                     Execute FIRST bounded waypoint, re-observe
\`\`\`

**重点**：稠密目标场保留多峰的不确定性；已观测视觉只更新被实际看见的候选，
“看不见”不代表“目标不存在”。静态名称—轮廓绑定不需要监督训练；
空间关系的图注意力、目标概率与轨迹头联合学习。地图坐标统一用
(东向 x, 北向 y) 的公有地图范围归一化位置。

模型设计与输入输出、损失、因果约束详见
[ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 安装

建议 Python 3.11、PyTorch 2.3+。实际 GPU 训练推荐先在本地 5090 检查 CUDA/PyTorch 支持。

\`\`\`bash
pip install -e '.[dev]'
# 可选：实际 BERT/SigLIP2 需要 Hugging Face Transformers 及本地预训练模型
pip install -e '.[foundation]'

pytest -q
python -m graphnav.train --smoke --epochs 1 --steps 2 --device cpu \
  --grid-size 8 --max-landmarks 4 --top-k 4 --hidden 32 --text-dim 32 \
  --waypoints 4 --output runs/smoke/checkpoint.pt
\`\`\`

Smoke 使用**合成张量**（仅测试张量、反向传播、checkpoint），不等于实际训练。

## CityNav 数据训练

先将现有 HETT 工程中仅作为数据元信息的地图 bounds 导出（新网络运行时不依赖 HETT）：

\`\`\`bash
python scripts/export_bounds.py \
  --mapdata /path/to/HETT/multiagent/mapdata.py \
  --bounds-out data/citynav_bounds.json \
  --ground-out data/ground_levels.json

python scripts/prepare_citynav.py \
  --objects /path/to/data/cityrefer/objects.json \
  --processed /path/to/data/cityrefer/processed_descriptions.json \
  --trajectories /path/to/data/processed_citynav/citynav_train_seenall.json \
  --bounds data/citynav_bounds.json --ground-levels data/ground_levels.json \
  --output data/graphnav_train_seen.jsonl

python -m graphnav.train --data data/graphnav_train_seen.jsonl \
  --epochs 10 --batch-size 16 --device cuda --output runs/graphnav/train.pt
\`\`\`

双卡（DDP）：
\`\`\`bash
torchrun --standalone --nproc_per_node=2 -m graphnav.train \
  --data data/graphnav_train_seen.jsonl --batch-size 16 --epochs 10 \
  --output runs/graphnav/ddp.pt
\`\`\`

视觉分支需要**因果历史 RGB** 训练样本；不能直接将全域正射图传给模型：

\`\`\`bash
pip install -e '.[foundation]'
pip install rasterio
python scripts/prepare_citynav.py \
  --objects data/cityrefer/objects.json \
  --processed data/cityrefer/processed_descriptions.json \
  --trajectories data/processed_citynav/citynav_train_seenall.json \
  --bounds data/citynav_bounds.json --ground-levels data/ground_levels.json \
  --rgbd-dir /path/to/data/rgbd --observations-out data/observed_train \
  --output data/graphnav_train_rgb.jsonl

python -m graphnav.train --data data/graphnav_train_rgb.jsonl \
  --use-vision --epochs 10 --batch-size 16 --device cuda \
  --output runs/graphnav/visual.pt

python -m graphnav.evaluate --data data/graphnav_val_unseen.jsonl \
  --checkpoint runs/graphnav/train.pt --device cuda
\`\`\`

转换器只会为每个训练时刻暴露它过去及现在实际观察的局部视角。
评估脚本仅统计 Goal R@K/20m 和 Top1 NE，不能代替闭环 SR/SPL。

这是初始**开放式监督训练**，而非已完成的在线 imitation rollout。
对比实验应在真实 CityNav 闭环 evaluator 接入后报告 SR/SPL/OSR/NE；
请参见 [DATA.md](docs/DATA.md) 和 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 代码

| 位置 | 作用 |
|---|---|
| graphnav/geometry.py | 世界坐标、轮廓采样、到多边形真实边界的有符号距离 |
| graphnav/graph.py | 纯 PyTorch Edge-GATv2、地标/UAV 稀疏图 |
| graphnav/model.py | Dense Goal Field + Top-K Proposal Graph + 多模态 Waypoints |
| graphnav/memory.py | 只累积真实 RGB 的世界坐标图像记忆 |
| graphnav/foundations.py | 冻结 BERT、SigLIP2、RGB Patch 缓存 |
| graphnav/runtime.py | 模型在线推理接口（外部仿真器提供观测，不提供 GT） |
| graphnav/data.py | CityNav JSONL 数据集、真实人类轨迹监督 |
| graphnav/losses.py | Gaussian 目标分布监督、候选排序、路径与 Stop 损失 |
| graphnav/train.py | 独立训练 CLI；BF16、DDP、Smoke |
| scripts/prepare_citynav.py | 从 CityNav 原始 JSON 转换为独立训练数据 |
| tests/ | 几何方向、视觉信息泄漏、数据绑定、梯度与张量测试 |

## 研究原则

- **零未来图像泄漏**：不从 CityNav 全域正射图裁剪尚未观察过的目标。
- **零目标标签泄漏**：GT goal / human future 仅进入训练 Loss，绝不进入模型 Forward 或在线 Runtime。
- **地图名称不由像素猜测**：确定性 name-ID-contour 绑定；模糊同名不强行选最近 GT。
- **多峰目标置信度保留**：不能只评估已入 Top-K 的候选；独立监督全部 28×28 单元。
- **Stop 不可直接部署**：需另做独立精度校准；在线接口默认只输出 stop_probability。

MIT-style research baseline: 无法替代 CityNav 原数据或第三方 pretrained 权重的许可证。
