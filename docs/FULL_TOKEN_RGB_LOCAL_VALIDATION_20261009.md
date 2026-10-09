# A0–A5 本地验证结果（2026-10-09）

上游分支 `2027-CVPR/full-token-geometry-observed-rgb`，基线 Commit `4e0e04de7a4dc48afa3a8b95317bd62c687bc0ca`。本次验证及修复保存在 `validation/full-token-rgb-local-20261009`，不合并 main。

A5 已完成 5 个 Epoch；按用户最新指令停止新增 batch 测试和进一步原因调查，记录已完成的结果。以下均为离线目标定位指标，不是导航 SR/SPL。未训练 Trajectory/Stop，未运行闭环。

## 最终统一评估

所有模型使用相同去重后的 Query：Val Seen 2,426 条，Val Unseen 2,679 条。命中半径 20m；Recall 用百分比，NE 用米；汇总及逐条输出均直接使用 Top-8 排序。附加 Top-20 仅作诊断，不能代替 R@8。

| 模型 | Seen R@1 | Seen R@4 | Seen R@8 | Seen NE | Unseen R@1 | Unseen R@4 | Unseen R@8 | Unseen NE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 Building Referenced Prior | 10.47 | 22.18 | 28.03 | 122.89 | 12.02 | 25.38 | 36.24 | 128.51 |
| A1 仅全类别 Prior（附加对照） | 15.21 | 35.78 | 45.47 | 59.77 | 15.01 | 33.71 | 43.60 | 69.30 |
| A1 Prior + 显式几何规则 | 15.75 | 37.18 | 47.07 | 59.01 | 15.83 | 34.68 | 46.06 | 67.66 |
| A2 BERT Token Cross-Attention | 31.99 | 53.17 | 65.54 | 40.73 | 11.31 | 25.83 | 38.52 | 71.85 |
| A3 候选–地标相对几何 | 40.68 | 62.57 | 73.50 | 31.49 | 23.78 | 39.38 | 50.88 | 48.42 |
| A4 Typed GATv2（batch 64） | 37.63 | 59.93 | 70.98 | 33.79 | 20.12 | 36.62 | 48.00 | 52.60 |
| A5 已观测 RGB 视觉残差（batch 16） | 36.89 | 60.22 | 71.35 | 33.79 | 20.46 | 37.40 | 48.11 | 52.54 |

A3 的 Unseen R@8 最高（50.88%）。最终 A4 比规则基线高 1.94 个百分点，但低于 A3；本轮结果不支持“GAT 优于候选显式几何”。A5 比 A4 的 Unseen R@8 高 0.11 个百分点（净增 3 条命中），不能作为稳定增益证据。

训练配置并非完全一致：A2/A3 是 batch 16、workers 0、文本上限 96；A4/A5 是文本上限 512，分别 batch 64/16、workers 4。旧 96 上限截断了 11 条训练、3 条 Seen Query，Unseen 无截断。因此该最终表是已选配置比较，不能将 A3→A4 差异全部归因于 GAT 架构。原始 batch-16 A4 对照已保留在 `runs/revised_static_a0_a4/`。

Checkpoint 仅按 Val Seen R@8 选择：A2 epoch 5，A3 epoch 4，A4 batch-64 epoch 4，A5 epoch 5。batch-128 的 Unseen 指标未用于选择配置。最终评估 batch 16，workers 4；加载每个 Checkpoint 的原文本截断设置。

## 训练记录

单卡 RTX PRO 6000 Blackwell Max-Q 96GB。训练头和视觉编码使用 BF16 autocast；冻结 BERT 使用 FP32。表内耗时是训练循环累计时间，不含验证、模型下载、数据转换和已中止试跑。首轮包含当轮冻结特征/几何缓存建立；跨模型缓存预热程度不同，不能直接解释为架构速度比。显存是 PyTorch 峰值 allocated，非 nvidia-smi 全进程占用。

| 模型 | batch / workers | Epoch 数 | 累计训练秒 | 最后一轮 samples/s | 峰值 allocated GiB |
| --- | --- | ---: | ---: | ---: | ---: |
| A2 BERT Token Cross-Attention | 16 / 0 | 5 | 349.30 | 1338.89 | 0.53 |
| A3 候选–地标相对几何 | 16 / 0 | 5 | 148.16 | 726.16 | 1.28 |
| A4 Typed GATv2（batch 64） | 64 / 4 | 5 | 227.33 | 681.42 | 3.83 |
| A5 已观测 RGB 视觉残差（batch 16） | 16 / 4 | 5 | 745.33 | 561.57 | 2.37 |

A5 五轮共 745.33 秒（12 分 25 秒）。仅训练视觉残差，原 A4 和 SigLIP2 均冻结；Checkpoint 中逐一对比 70 个非视觉基础模型 Tensor，完全一致。

| A5 Epoch | Loss | 训练秒 | samples/s | 峰值 GiB | Seen R@1 % | Seen R@4 % | Seen R@8 % | Seen NE m |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.938223 | 549.34 | 38.53 | 2.37 | 37.51 | 59.85 | 70.94 | 33.87 |
| 2 | 3.937469 | 64.78 | 326.76 | 1.84 | 37.43 | 59.98 | 71.06 | 33.67 |
| 3 | 3.936941 | 53.50 | 395.66 | 1.84 | 37.10 | 60.14 | 70.94 | 33.84 |
| 4 | 3.936221 | 40.02 | 528.92 | 1.84 | 36.93 | 60.22 | 71.10 | 33.67 |
| 5 | 3.935551 | 37.69 | 561.57 | 1.84 | 37.14 | 60.22 | 71.31 | 33.76 |

训练期间的 Seen 指标采用当时增量建立的冻结特征缓存；最终表采用统一评估缓存。不同 BERT padding/batch 计算导致少量 BF16 排名边界 Query 有差异，不能从多次评估中挑最高值。最终统一评估已保存 BERT（96/512）及 SigLIP2 特征缓存，以固定此结果的输入特征。

A4 batch-64 与 batch-128 均完成五轮，累计分别 227.33 / 225.50 秒；Seen R@8 71.06% / 68.80%，按 Seen 选择 64。batch-256 按用户指令在两轮后停止，不纳入配置选择；后续不再扩大 batch，也未做 A5 batch sweep。原始扫参结果见 `a4_batch_sweep.csv/json`。

## 数据和观测覆盖

使用本地 SBF 复现目录的 Revised CityNav 原文件，逐项 SHA256 核验；未沿用原先 Original CityNav 的指标直接比较。真实 split 文件名为 `citynav_train_seen.json`、`citynav_val_seen.json`、`citynav_val_unseen.json`。

- 去重依据 map+instruction，保留首条源记录，不以目标坐标选样。最终 Train Seen 21,166 / Val Seen 2,426 / Val Unseen 2,679，训练与验证、Seen 与 Unseen Query 交集均为 0。源数据包含同文本不同标注：训练 59 组、Unseen 7 组，详见去重 JSON；未按 GT 消除歧义。
- 坐标采用本地米制地图，x 东、y 北；RGB 像素 y 反向映射。PNG 与对应 TIF 维度一致，公开 bounds 与 TIF bounds 最大差 0.09375m，所有保留目标在 bounds 内。地标名称绑定对象原始轮廓，未凭空生成几何。
- 有真实 TrafficRoad / Footpath 等封闭区域轮廓，可用于道路表面地标；没有验证过的道路中心线/拓扑。因此未提供 roads-json、未虚构道路网络。
- 仅初始观测 step=0，不能使用候选未来视角。1 条训练样本初始相机高度不高于地面，保留 Query 并显式标记 RGB 不可用，视觉残差为零。
- Forward 输入剔除 target_xy 等 GT 字段；A4/A5 静态输入及标签签名一致。签名包含标签用于完整性检查，不送入网络。

| 覆盖项 | Val Seen | Val Unseen |
| --- | ---: | ---: |
| 语言关系有成功解析并绑定的 Query | 17.89% | 20.98% |
| 道路表面地标 Query | 69.91% | 59.13% |
| A5 有有效 RGB 的 Query | 34.30% | 32.81% |
| A5 有有效 RGB 的候选 | 22.67% | 20.54% |

关系覆盖率表示至少一个成功 grounded 关系的 Query 占比，并非人工标注的解析准确率。RGB 候选覆盖率的分母是每条 Query 的 Top-20 候选。无有效 RGB 的 Seen 1,594 条、Unseen 1,800 条 Query，A5 的 NE 和 R@1/4/8 均与 A4 完全一致。

已完成的配对统计：A5−A4 Seen R@8 +0.371pp，Query bootstrap 95% CI [-0.082, 0.824]pp；Unseen +0.112pp，CI [-0.261, 0.485]pp。地图聚类 CI 也包含 0；Unseen 仅 4 张地图，单随机种子，不能证明稳定提升。未追加原因调查。

已完成的 GAT 推理敏感性检查：保留同一权重、删除非自环边后，Seen R@8 从 70.98% 降到 69.50%；Unseen 从 48.00% 变为 48.08%，NE 从 52.60m 变为 55.04m。这是推理扰动，不是独立重训的架构消融。

## 已修复问题与验证

1. Revised split 文本和 objects 旧文本可能不同：转换器采用 split 内权威 instruction。去重后刷新输出 SHA256 和记录数，防止 provenance 失效。
2. 内部点到轮廓距离原来为零：相对几何现在计算闭合边界的负内部深度；CPU packed FP16 几何在模型入口对齐 dtype。
3. GEOS 精确距离加速真实多顶点轮廓，保留无 Shapely/无效几何原路径；取消 8,192 项几何缓存和 12,000 项文本缓存逐轮驱逐，启用 persistent workers。
4. A5 不可用 RGB 样本保留，候选中心严格空覆盖筛除；冻结 A4 场和视觉特征缓存。视觉空 batch 使用真实 vision hidden_size，SigLIP2 processor 使用明确 slow 设置。
5. 按实际样本加权 Loss；逐轮保存 last checkpoint、optimizer、history、吞吐量、显存和 Seen 指标，best checkpoint 附配置及静态签名。
6. 统一评估逐条 Top-8 与汇总 Top-8 的同分排序；Top-20 仅用于独立诊断。全部模型的逐条命中数和 NE 与汇总核对通过。冻结特征缓存持久化并校验模型 revision/precision/文本上限。

验证：`pytest -q` 全部 56 项通过，含 GAT→RGB CPU 端到端、冻结参数不变、篡改标签拒绝加载、signed boundary、旋转/非方形地图观测覆盖等价、特征缓存保存/加载协议、同分 logits 逐条指标一致性。`pip check` 通过，`ruff --isolated --select E9,F63,F7,F82` 通过，真实 SigLIP2 本地 smoke 输出 768 维单位向量。

## 保留的限制与产物

已经观察到首轮视觉渲染/裁剪/编码缓存开销，以及后续 CPU 数据搬运开销；显存远未耗尽。按最新指令只记录现有证据，不继续 profiling 或失败原因试验。

相对于 SBF，这条实验链仍没有完整九通道场景栅格（高度、全部场景地标、当前机位、已探索、历史轨迹），使用 BERT 而非 SigLIP 文本 Token；候选采用原始网格 Top-K，尚无 SBF 的 NMS/地标中心增补及独立候选选择 Transformer；监督也未完全复现 SBF 米制 Gaussian + ranking loss。A5 是冻结 A4 上的 pooled SigLIP2 残差，仅初始已观测视角。记录这些范围差异，不将当前结果等同于完整 SBF 复现。

代码/报告/CSV/JSON 提交 Git；权重、原始 Query 预测 JSONL、冻结特征缓存、RGB 帧和数据保存在本地，未将二进制权重和数据加入 Git。所有最终产物的路径、大小、SHA256 及选中 Epoch 见 `reports/full_token_rgb_local_20261009/artifact_manifest.json`。

- 最终 Checkpoint：`runs/final_a0_a5_evaluation/{tokens,geometry,gat,rgb}_best.pt`。
- A5 完整训练 Checkpoint/history：`runs/a5_from_a4_batch64/rgb_{best,last}.pt`、`rgb_history.json`。
- 最终 CSV/JSON、四个模型原始逐轮 history、训练表、数据审计、配对统计和已有拓扑检查：`reports/full_token_rgb_local_20261009/`。
- 逐条最终预测及 BERT/SigLIP2 特征缓存：`runs/final_a0_a5_evaluation/`。
- 原始日志：`runs/local_validation/`。

### 复评命令

在当前仓库根目录，使用已安装依赖的环境。以下命令加载已选权重及已保存缓存，不重训：

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false python -m graphnav.full_experiment \
  --train data/revised_rgb/graphnav_train_seen.jsonl \
  --val-seen data/revised_rgb/graphnav_val_seen.jsonl \
  --val-unseen data/revised_rgb/graphnav_val_unseen.jsonl \
  --dataset-version revised --match-mode fuzzy \
  --variants tokens geometry gat rgb --evaluate-only \
  --batch-size 16 --workers 4 --visual-cache-size 100000 \
  --output runs/final_a0_a5_evaluation
```

数据转换选项及原始训练链参见 [FULL_TOKEN_GEOMETRY_RGB.md](FULL_TOKEN_GEOMETRY_RGB.md)。实际本机 Python 为 `../2027-CVPR-static-geometry/.venv/bin/python`，须从本工作树运行并设置 `PYTHONPATH=.`。
