# UnifiedPGVB-V2

**Language-Grounded Goal Belief for Aerial Vision-and-Language Navigation**

**CityNav / CityRefer · Research implementation · 中文文档**

GraphNav 是面向城市级无人机视觉语言导航（VLN）的研究工程：利用**具名地标及其轮廓几何**，从语言指令推断全局目标位置，再通过有界控制器执行导航。项目包含在真实可见范围内提取 RGB/Depth 证据、累积历史记忆和更新目标信念的实验实现。

> **当前主方案：UnifiedPGVB-V2（集成几何+单视觉目标Belief）**。主定位网络为 `graphnav/unified_pgvb.py`：内部集成原A3几何定位、唯一SigLIP2/PGVB视觉评分和统一GoalField输出；训练前的联合融合层默认关闭。原Trajectory/Stop作为独立冻结控制器。**已完成实验分支上的真实训练与完整Val-Seen闭环：完整RGB方案SR 50.00%，关闭RGB的匹配消融SR 50.24%（各2470条，seed=42）。main保留集成入口，实验权重、运行配置和结果来源见下方链接；这不表示本次文档提交重新评测或合入全部实验代码。**请从 [V2架构说明](reports/unified_pgvb/README_zh.md) 和 [V2测试计划](docs/UNIFIED_PGVB_V2_TEST_PLAN_zh.md) 开始。

> **GraphNav-V1为历史正式基线，不再是主方案。** 已完成Val-Seen、Val-Unseen、Test-Unseen的完整评测：Seen SR/SPL 47.61%/46.00%，Val-Unseen 22.54%/21.82%，Test-Unseen 37.15%/35.81%。 完整准确数值以原始[基线报告](reports/graphnav_v1_stage1/GRAPHNAV_V1_STAGE1_REPORT.md)为准。原始代码完整冻结于 [archive/graphnav-v1-pre-unified-20261104](https://github.com/qiaoxu123/2027-CVPR/tree/archive/graphnav-v1-pre-unified-20261104)（`9098e254`），权重仍需外部按SHA核验。

[快速开始](#快速开始) · [已验证结果](#已验证结果) · [评估与复现](docs/REPRODUCIBILITY.md) · [代码导航](docs/CODE_MAP.md) · [研究报告索引](#研究报告与局限)

> **主方案更新（2026-11-07，仅设计与实验依据，未改变默认代码）：** 在 Geometry 的语言—地图输入端纳入「公开地标名称变体绑定」：仅对当前场景唯一、无歧义的地标简称、服务商前缀及单复数别名建立公开 Polygon 关联，保留整句原文和多实例歧义；基于原28×28 Goal Belief 采用固定局部 soft 位置解码，并以20m空间 NMS 建立分散的候选池。此机制是 Geometry 的输入适配和候选解码，不单列论文核心创新，视觉分支及 Controller 保持解耦。源实验分支 `experiment/v2-top1-public-binding-20261107` 取得完整 Val-Seen 初始 R@20m **50.65%**、NE **29.66m**、NMS Recall@5 **80.08%**（2470条，原始V2 Geometry与新解析/解码组合）；**不是闭环SR，不等同于当前默认运行时已达到这些数值**。新增的同权重、同Controller、2470条完整闭环单因素对照已验证：扩展绑定 **SR 48.54%**，旧绑定 **47.25%**（净增32条成功，+1.30个百分点），**NE 30.83m vs 33.31m**；这仍是关闭RGB的Geometry研究配置，不能与下面其他Checkpoint和历史标注绑定的RGB融合结果混排。详细集成边界见 [V2架构说明](reports/unified_pgvb/README_zh.md)。

## 当前主方案：UnifiedPGVB-V2

```text
原始导航指令 + 公开地标Polygon + 当前RGB
                 |
      GT-free结构化指令解析
                 |
          UnifiedPGVB-V2
       +---------+---------+
       |                   |
  A3 Geometry        单次SigLIP2 RGB
 (网络内部子模块)     PGVB视觉地标证据
       |                   |
       +----- 内部Belief融合 +
                 |
      28×28统一GoalField
   Top-K + 128维候选上下文
                 |
       冻结Trajectory / Stop
                 |
         无人机动作与新观测
```

main的基础接入支持具名地标导航；后续实验分支已完成完整CityNav原始目标的Geometry、Relation及RGB匹配消融。Relation与RGB的额外收益尚未得到支持，不能将地标导航高SR代替完整CityNav成绩。训练或评测前先运行 [V2测试计划](docs/UNIFIED_PGVB_V2_TEST_PLAN_zh.md)。旧GraphNav-V1架构图仅作为历史基线：

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

**以下为原GraphNav-V1的历史说明，并非V2当前默认前向。**

**区别与边界**

- **A3（主线）**：逐候选对名称、地标几何和语言 Token 进行注意力匹配；没有地标间的 GAT 消息传递。
- **A4（消融）**：A3 的兼容结构上增加两层 Edge-GATv2；已有实验**未证明**图关系节点提供稳定的额外语义推理收益。
- **视觉 / History（实验性）**：只接受当前及历史已经真实观察到的图像区域。未见区域保持“未知”，不等于负样本；视觉特征的额外导航增益尚待严格消融验证。
- **控制与指标**：逐步重规划、限制单步位移与高度；CityNav benchmark 的 SR 采用二维目标距离，不能宣称三维避障或真实第一视角相机能力。

## 已验证结果

### 公开名称绑定：同模型全量闭环对照（独立实验协议）

冻结同一 V2 Geometry 与 Stage1 Controller，关闭RGB/OSM，2470/2470条 HETT Refined Val-Seen、seed=42，唯一改动为公开名称解析与绑定。

| 名称绑定 | SR | OSR | SPL | NE (m) | nDTW |
|:--|--:|--:|--:|--:|--:|
| 旧绑定 | 47.25% | 54.41% | 45.91% | 33.31 | 29.51% |
| **扩展绑定** | **48.54%** | **55.43%** | **47.15%** | **30.83** | **29.89%** |

扩展绑定纠正53条、破坏21条，净增32条成功（SR +1.30个百分点）；2225条轨迹完全相同。所有任务以策略停止，扩展组51.46%的终止位置在目标20m外，该比例**不能单独证明Stop分类器误判**。这是闭环实测（不是离线 R@20m 50.65%，也不是开启RGB的UnifiedPGVB完整系统）；与下表的其他Checkpoint/地标绑定协议不可直接混同比较。来源：[实验报告](https://github.com/qiaoxu123/2027-CVPR/blob/4e0758e493acba8ced21ffdf61ca981a0d33fed3/reports/unified_pgvb/top1_public_binding/navigation_20261107/REPORT_zh.md)。当前仅融合研究方案和证据，**尚未修改main默认Runtime、权重或训练脚本**。

### 论文主方案：UnifiedPGVB-V2（完整Val-Seen）

以下来自同一轮实验，各组均覆盖2470/2470条原始CityNav任务，seed=42；失败、无候选及回退任务均计入分母。Controller、Stop和成功阈值一致。后三组共享训练后的Geometry。

| 方法 | SR | OSR | SPL | NE (m) | nDTW | 决策耗时 (ms/步) |
|:--|--:|--:|--:|--:|--:|--:|
| 训练前Geometry | 47.77% | 55.51% | 0.4633 | 29.198 | 0.3029 | 11.33 |
| 训练后Geometry | **50.24%** | 57.33% | **0.4875** | 28.465 | **0.3051** | 11.09 |
| Geometry + Relation（关闭视觉） | **50.24%** | 57.33% | **0.4875** | **28.463** | **0.3051** | 24.64 |
| UnifiedPGVB-V2：Geometry + Relation + RGB | 50.00% | **57.41%** | 0.4848 | 28.522 | 0.3050 | 34.99 |

视觉消融：RGB纠正2条、破坏8条，净−6条（SR −0.24个百分点）；Relation相对训练后Geometry纠正/破坏为0/0。当前最好SR来自关闭视觉的版本，不能据此声称视觉带来提升。耗时来自共享GPU，单seed结果不代表多种子稳定性；本轮未完成Val-Unseen/Test评估。

结果固定来源：[中文实验报告](https://github.com/qiaoxu123/2027-CVPR/blob/4d195f841ba9c3a561555ca75bcf44bfea79b6cc/reports/unified_pgvb/full_relation_s42_20261105/REPORT_zh.md)、[逐组原始指标与分层](https://github.com/qiaoxu123/2027-CVPR/blob/4d195f841ba9c3a561555ca75bcf44bfea79b6cc/reports/unified_pgvb/full_relation_s42_20261105/metrics.json)。47.37%是另一轮全文公开绑定基线，不能与本表混作同一次消融；62/64是具名地标专项结果。

### 正式全量闭环结果（GraphNav-V1 C Epoch1）

| Split | Episodes | SR | SPL |
|:--|--:|--:|--:|
| Val-Seen | 2,470 | **47.61%** | **46.00%** |
| Val-Unseen | 2,697 | **22.54%** | **21.82%** |
| Test-Unseen | 5,281 | **37.15%** | **35.81%** |

配置固定：`--model graphnav-v1`，`--frozen-final-candidate` 按已冻结 Seen 选择清单验证，`--stop-threshold 0.3`、`guarded-blend`，没有主动观察、新 GAT、关系辅助损失。**本地已有评估是权重对应源版本上的实测，不等于此集成提交重新进行了 GPU 全量重放。**

### 历史静态与控制基线（独立保留）

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

> **协议区分：** 上表 Seen128 仅用于历史诊断，不与正式全量指标混排。A3 B0 的正式全量 Seen/Unseen 及 V1 的指标参见 [V1 阶段报告](reports/graphnav_v1_stage1/GRAPHNAV_V1_STAGE1_REPORT.md)。SBFNav 是本地论文方法独立复现，非作者官方权重；视觉/History 尚无稳定因果净收益证据。

## 快速开始

### 1. 环境

推荐 Linux、Python 3.10/3.11、匹配本机 CUDA 的 PyTorch；实际 GPU 训练使用过 BF16。CPU CI **只测试单元逻辑及合成 Smoke**，不下载 CityNav 原始数据、BERT/SigLIP2 或研究 Checkpoint。

```bash
git clone https://github.com/qiaoxu123/2027-CVPR-Dev.git
cd 2027-CVPR-Dev

# 请先按本机 CUDA/驱动安装兼容的 PyTorch；再安装项目依赖
python -m pip install -e '.[dev,data,foundation]'
python -m pip install pyyaml

python -m pytest -q
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

**当前限制：** GraphNav-V1 的来源报告及对应代码已并入 main，但完整 Backbone 与最终 `.pt` 权重需要从用户本地资源取得，且本次集成未重新执行全量 GPU 回放。视觉/History 消融尚未证明稳定净收益；真实透视相机、遮挡与三维避障不属于已验证能力。

## 引用与许可

若使用 CityNav 数据，请引用 [CityNav 项目提供的论文及许可信息](https://github.com/water-cookie/citynav)。本仓库尚无正式 GraphNav 论文 BibTeX 或已核实的再分发许可；请不要把研究代码名称冒充已发表论文。第三方数据、模型与复现工程的权利归各自所有。
