# GraphNav

**Language-Grounded Goal Belief for Aerial Vision-and-Language Navigation**

**CityNav / CityRefer · Research implementation · 中文文档**

GraphNav 是面向城市级无人机视觉语言导航（VLN）的研究工程：利用**具名地标及其轮廓几何**，从语言指令推断全局目标位置，再通过有界控制器执行导航。项目包含在真实可见范围内提取 RGB/Depth 证据、累积历史记忆和更新目标信念的实验实现。

> **研究状态（2026-10-10）**：最强已验证的静态定位骨干是 **A3 Geometry**，不是 GAT。A3 Epoch14 的 Val Seen 静态 R@1/20m 为 **48.47%**；固定 Seen128 闭环 SR 为 **48.44%**。这**不是全量 Seen 或 Unseen 的最终性能**。完整视觉/History 联合方案仍处研究与训练阶段，不能作为已完成的性能提升宣称。本仓库并非已接收的 CVPR 论文官方实现。

[快速开始](#快速开始) · [已验证结果](#已验证结果) · [评估与复现](docs/REPRODUCIBILITY.md) · [代码导航](docs/CODE_MAP.md) · [研究报告索引](#研究报告与局限)

## 方法概览

```text
语言指令 + 公共地图（具名地标、真实 Polygon）
                   |
       冻结 BERT Token / 名称特征
                   |
        A3 Geometry（静态定位）
       地标轮廓 + 候选相对几何
                   |
        28 × 28 Goal Belief B0
                   |
       Top-K 位置及 128-D Context
                   |
       +-----------+-----------+
       |                       |
 已验证：原 Trajectory/Stop   实验性：观察到的 RGB/Depth
 + Guarded / Bounded 控制     SigLIP2 + History 信念残差
       |                       |
       +-----------+-----------+
                   |
          当前 Pose → 下一 Waypoint
                   |
          执行一步、重新观测、重规划
```

**区别与边界**

- **A3（主线）**：逐候选对名称、地标几何和语言 Token 进行注意力匹配；没有地标间的 GAT 消息传递。
- **A4（消融）**：A3 的兼容结构上增加两层 Edge-GATv2；已有实验**未证明**图关系节点提供稳定的额外语义推理收益。
- **视觉 / History（实验性）**：只接受当前及历史已经真实观察到的图像区域。未见区域保持“未知”，不等于负样本；视觉特征的额外导航增益尚待严格消融验证。
- **控制与指标**：逐步重规划、限制单步位移与高度；CityNav benchmark 的 SR 采用二维目标距离，不能宣称三维避障或真实第一视角相机能力。

## 已验证结果

**离线静态定位**：Val Seen 2,426 条去重 Query，阈值 20m；**闭环**：固定 Val Seen 128 Episodes，使用已有 Trajectory/Stop、Guarded-Blend、有界动作，20 步上限。

| 模型 / 协议 | R@1/20m | R@4/20m | R@8/20m | SR | SPL |
|:--|--:|--:|--:|--:|--:|
| A3 Seed1 Epoch5 · 静态 + Seen128 | 42.54% | 64.43% | 75.47% | 40.63% | 40.09% |
| **A3 Seed1 Epoch14 · 静态 + Seen128** | **48.47%** | **68.63%** | **78.03%** | **48.44%** | **47.41%** |
| A4 Full GAT Epoch5 · 静态 + Seen128 | 41.92% | 63.48% | 75.02% | 43.75% | 42.93% |

其他不可混用的协议：

| 已归档系统 | 数据集 | SR | SPL |
|:--|:--|--:|--:|
| 旧 A3 + 有界目标移动 | Val Seen **全量 2,470 Episodes** | 40.89% | 39.77% |
| 本地论文方法复现 SBFNav Epoch14 | Val Seen **全量 2,470 Episodes** | 43.52% | 36.12% |
| 同一 SBFNav 复现 | Val Unseen **全量 2,697 Episodes** | 21.51% | 15.89% |

> **不可直接跨行排序**：新 A3 Epoch14 仅完成 Seen128 闭环，不可与 SBFNav 全量 SR 作严格比较。原 A3/A4、关系辅助监督、GAT 反事实及置信区间见 [受控实验报告](docs/A3_RELATION_GAT_CONTROLLED_20261010.md)。本地 SBFNav 是论文方法的**独立复现**，非论文作者官方权重。完整视觉/History 模型尚未通过全量 Seen/Unseen 验证。

## 快速开始

### 1. 环境

推荐 Linux、Python 3.10/3.11、匹配本机 CUDA 的 PyTorch；实际 GPU 训练使用过 BF16。CPU CI **只测试单元逻辑及合成 Smoke**，不下载 CityNav 原始数据、BERT/SigLIP2 或研究 Checkpoint。

```bash
git clone https://github.com/qiaoxu123/2027-CVPR-Dev.git
cd 2027-CVPR-Dev

# 请先按本机 CUDA/驱动安装兼容的 PyTorch；再安装项目依赖
python -m pip install -e '.[dev,data,foundation]'
python -m pip install pyyaml

pytest -q
python -m graphnav.train --smoke --epochs 1 --steps 2 --device cpu \
  --grid-size 4 --max-landmarks 3 --top-k 3 --hidden 32 \
  --text-dim 16 --waypoints 3 --batch-size 2 \
  --output runs/smoke/checkpoint.pt
```

**Smoke ≠ CityNav 训练或闭环导航。** 所有正式命令都需要本地授权取得的数据、预训练模型与匹配的 Checkpoint。

### 2. 数据与权重

项目基于 [CityNav 官方数据/环境](https://github.com/water-cookie/citynav)。数据及第三方预训练权重**不随本仓库分发**；使用前请遵守各自许可和数据集划分。

准备本地路径：

- `SBF_SOURCE`：经验证的本地 CityNav/SBF 复现源码目录（提供 `sbfnav` 导航环境依赖）。
- `SBF_CONFIG`：与该数据版本对应的 SBF 配置。
- `CITYNAV_DATA`：经验证的 CityNav 数据根目录。
- `A3_BEST`：A3 Seed1 Epoch14 的 `best.pt`。
- `REFERENCE_A3`、`HEAD_BEST`：原已训练控制器对应的静态骨干和 Trajectory/Stop 权重。

权重不会从 README 中的服务器绝对路径自动下载。SHA、来源和完整参数见 [评估与复现](docs/REPRODUCIBILITY.md)。

### 3. 复现强基线（闭环）

```bash
python scripts/evaluate_closedloop.py \
  --model static-transfer \
  --sbf-source "$SBF_SOURCE" \
  --config "$SBF_CONFIG" \
  --checkpoint "$A3_BEST" \
  --reference-a3-checkpoint "$REFERENCE_A3" \
  --head-checkpoint "$HEAD_BEST" \
  --head-execution guarded-blend \
  --stop-threshold 0.3 \
  --data-root "$CITYNAV_DATA" \
  --split val_seen --max-episodes 128 \
  --output runs/eval_a3_e14_seen128
```

要跑全量 Seen，请使用全新输出目录并移除 `--max-episodes 128`。**首次复现请先确认 Checkpoint SHA、128 个 Episode 清单及单步控制一致性。** 任何 Unseen 评估都不能反向用于选择模型或调阈值。

训练、评估、不同运行协议和临时数据的完整说明见 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

## 代码结构

```text
graphnav/          核心模型、地图几何、静态热图、视觉/History、轨迹与 Runtime
scripts/           数据准备、训练、评估、诊断 CLI（详见 scripts/README.md）
tests/             单元测试及防泄漏 / 兼容性回归
docs/              使用文档、协议和历次研究结论
reports/           实验指标、消融及审计记录（历史证据，非代码入口）
configs/           已归档实验环境约束
.github/workflows/ CPU CI
```

快速定位关键文件：

| 需求 | 推荐入口 |
|:--|:--|
| 静态目标场 A3 / A4 | `graphnav/full_localization.py` |
| 地标关系图 / GAT | `graphnav/semantic_graph.py` |
| 强基线权重迁移 | `graphnav/static_transfer_runtime.py` |
| RGB 历史与残差信念 | `graphnav/dynamic_belief.py`、`graphnav/joint_active.py` |
| 轨迹 / Stop | `graphnav/trajectory_policy.py` |
| 统一闭环评估 | `scripts/evaluate_closedloop.py` |
| 已有联合训练实验 | `scripts/train_joint_active.py`（见下方警告） |

详见 [代码入口导航](docs/CODE_MAP.md)。

## 研究报告与局限

- [A3 延长训练 / 关系监督 / GAT 受控消融](docs/A3_RELATION_GAT_CONTROLLED_20261010.md)：当前静态强基线与权重 SHA。
- [共享闭环评估协议](docs/SHARED_CLOSEDLOOP_PROTOCOL_20261009.md)：Episode、评价维度、动作约束。
- [已观测 RGB 与 SBF 输入差异](docs/GRAPHNAV_SBF_INPUT_PROCESSING_AUDIT_20261010.md)：视觉对齐、冻结模型和负对照。
- [联合训练 Epoch1 问题及 Stop 修正](docs/JOINT_EPOCH1_CONTROL_FIX_AND_ABLATIONS.md)：**旧联合训练会重新初始化控制器，不可用来替代强基线**。
- [复现及结果索引](docs/REPRODUCIBILITY.md)：如何区分正式指标与开发性诊断。

**当前未解决：** 新 A3 的完整 Val Seen/Unseen 闭环结果、真实 RGB 对候选重排的因果增益、可信历史纠错、真实透视相机/遮挡和三维安全避障。未经训练或评估的模块不以结果形式展示。

## 引用与许可

若使用 CityNav 数据，请引用 [CityNav 项目提供的论文及许可信息](https://github.com/water-cookie/citynav)。本仓库尚无正式 GraphNav 论文 BibTeX 或已核实的再分发许可；请不要把研究代码名称冒充已发表论文。第三方数据、模型与复现工程的权利归各自所有。
