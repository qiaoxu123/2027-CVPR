# Static Semantic–Geometric Localization 验证报告（2026-10-09）

基准分支：`2027-CVPR/semantic-geometry-localization`；基准提交：`4d6c81c2395505fb4473a08383d0476576acd776`。本地验证分支：`validation/static-geometry-20261009`。未合并 main。

## 结论

S2 在 Val Seen 上明显提升，但 Val Unseen 的 R@1 略低于 S1，R@8 的提升仅 0.68 个百分点且配对 95% 区间包含 0，不能证明其稳定优于规则基线。NE 降低不能替代 20m Recall。图边有局部作用，但大部分 Seen 提升来自不依赖图拓扑的文本/位置残差。

## 实验约束与环境

- 单卡 NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition，96GB；batch size=16，GAT BF16 autocast；BERT FP32、冻结、eval。
- Python 3.10.22、PyTorch 2.8.0+cu128、Transformers 4.57.6。隔离 `.venv`；依赖检查无冲突。版本约束位于 `configs/static_geometry_rtx6000_constraints.txt`。
- 全局 `/etc/gitconfig` 将 GitHub 重定向到 ghfast 镜像，鉴权失败；拉取时禁用系统 Git 配置并使用已有 gh 凭据，未修改全局配置。
- 已有其他 GPU 训练进程 PID 366555 被保留。耗时不是独占 GPU 的严格硬件基准。PyTorch 分配/保留显存不等于整卡 nvidia-smi 占用。
- S0/S1 无训练；S2 hidden=128、4 heads、max_landmarks=16、max_relations=8、max_roads=0、28×28 全域网格、sigma=25m、lr=2e-4、seed=0、5 Epoch。
- 不训练 RGB、Depth、Trajectory、Stop，不输入 UAV pose/future，不执行闭环，不报告 SR/SPL。

## 数据、坐标与绑定

原始数据：`/home/20260922_1/Workspace/DATA/citynav/extracted/data`。独立 JSONL：仓库 `data/graphnav_{train_seen,val_seen,val_unseen}.jsonl`。转换使用 `--static-only`，每个源注释仅生成静态指令、地标和地图元数据，目标仅为标签。

| Split | 原始 Query | 文本去重后 | 最终 Query | 建筑地标覆盖 | 已锚定关系覆盖 |
|---|---:|---:|---:|---:|---:|
| train_seen | 22002 | 21303 | 21289 | 29.25% | 6.96% |
| val_seen | 2498 | 2454 | 2454 | 28.97% | 6.72% |
| val_unseen | 2826 | 2808 | 2808 | 27.85% | 6.13% |

按精确 `(map_name, instruction)` 去重，按源文件顺序保留首条，不使用 GT 坐标挑选。Train 与 Val Seen 的 14 个重复文本 Query 从 Train 移除；三个 split 之间无 Query ID 或 map+instruction 重叠。Val Unseen 与训练地图分离。所有 S0/S1/S2 均使用同一验证记录和分母。
重复源文本存在标签冲突：Train 59 条（最大 142.69m），Val Seen 0 条，Val Unseen 7 条（最大 6.97m）。清单及固定保留规则已保存，未按标签距离决定去留。
公共 MAP_BOUNDS 来自现有 HETT mapdata 的元数据常量；x/y 使用地图 world XY 米制轴，x east/y north，方向规则与 UAV yaw 无关。正射 TIFF 的 x 像素向右、y 世界坐标向上；抽查公共边界与 TIFF 边界偏差约 0.05–0.08m，未从目标反推地图范围。所有目标均在地图内，physical_map_sizes 与 bounds 差值一致。逐个 id/name 校验绑定轮廓回投原始 CityRefer，最大误差 0.000172m。同名建筑保留多个 ID，OR 聚合，不选择 GT 最近对象。
原绑定会把 TrafficRoad/Footpath/Bridge 等任意同名轮廓作为建筑；现已按源 `object_type=Building` 过滤。CityRefer 有真实、同坐标系的道路面轮廓，但没有可验证的道路中心线与拓扑输入；没有把面边界当作道路中心线，也没有虚构道路。运行时道路实体覆盖 0%，真实道路关系覆盖率为 N/A。
Val Seen 有 66.14% 的指令含道路相关词，Val Unseen 为 57.26%；Building-only 去除了这些道路几何信息，这是主要覆盖损失。严格名称匹配保留了已知不足，如 “One Stop Building” 与地图 “One Stop”、 “BMTR building” 与 “BMTR”；未用 GT 引入猜测别名。
关系解析统计是“至少一条可靠、已命名锚点关系”的 Query 覆盖率，不是语言解析 precision/recall。Val Seen 的 1189 条关系触发词 Query 中仅 165 条得到锚定关系；Unseen 1409 条中仅 172 条。不能把图像中的 left/right/front/behind 无条件解释为世界东/西/南/北。
Val Seen 目标类型包括 Building 1058、Car 1209、Parking 56、Ground 131；Val Unseen 为 1170、1068、152、418。许多汽车/停车行列描述需要当前阶段缺失的信息。BERT 96-token 截断仅影响 Train 11 条、Seen 3 条、Unseen 0 条，不是主要瓶颈。

## 统一指标

R@k：前 k 个全域网格中心中任一个距离目标 ≤20m；NE：top1 的物理距离。只在 Val Seen R@8 上选择 Checkpoint，第 5 轮被选中；Val Unseen 不用于选模或调参。

| Split | 模型 | N | R@1/20m | R@4/20m | R@8/20m | NE(m) |
|---|---|---:|---:|---:|---:|---:|
| val_seen | S0 Prior | 2454 | 6.72% | 13.73% | 17.56% | 183.97 |
| val_seen | S1 Prior + Rules | 2454 | 6.56% | 14.06% | 17.73% | 183.99 |
| val_seen | S2 Prior + Rules + Typed GATv2 | 2454 | 12.31% | 23.55% | 31.91% | 103.71 |
| val_unseen | S0 Prior | 2808 | 5.48% | 11.08% | 15.60% | 205.12 |
| val_unseen | S1 Prior + Rules | 2808 | 5.77% | 11.25% | 15.49% | 205.19 |
| val_unseen | S2 Prior + Rules + Typed GATv2 | 2808 | 5.70% | 12.25% | 16.17% | 157.15 |

## 每轮训练记录

| Epoch | Loss | 训练(s) | samples/s | 分配峰值(GiB) | 保留峰值(GiB) | Val Seen R@8 | Val Seen NE(m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.9148 | 74.19 | 287.0 | 0.594 | 0.672 | 25.88% | 114.36 |
| 2 | 5.7268 | 15.48 | 1375.5 | 0.601 | 0.674 | 27.34% | 123.35 |
| 3 | 5.6022 | 15.56 | 1367.9 | 0.601 | 0.674 | 30.15% | 104.26 |
| 4 | 5.5150 | 15.61 | 1363.4 | 0.601 | 0.674 | 31.01% | 103.94 |
| 5 | 5.4527 | 15.82 | 1345.7 | 0.601 | 0.674 | 31.91% | 103.71 |

完整 5 Epoch 训练耗时 136.67s；每轮 Seen 验证总计 5.95s。不包括安装、数据准备、最终评估、诊断和写报告。首轮包括冷几何/BERT 编码；后续轮次复用静态几何与文本缓存。
Loss 从第 1 轮到第 2 轮下降，但 NE 从 114.36m 恶化到 123.35m，说明不能凭 Loss 判定效果。

## GAT 拓扑与泛化诊断

使用相同已训练权重，仅保留自环、移除非自环边做推理干预。此干预不是另行训练的无图基线，不能视为完全独立的架构消融。

| Split | S2 R@8 | 仅自环 R@8 | 拓扑增益(pp) | S2−S1 R@8 95% CI(pp) |
|---|---:|---:|---:|---|
| val_seen | 31.91% | 30.60% | 1.30 | [12.39, 15.85] |
| val_unseen | 16.17% | 15.31% | 0.85 | [-0.14, 1.50] |

5000 次配对 Query bootstrap；这是单 seed、当前 Query 样本上的不确定性，不等价于跨 seed 或地图簇的置信保证。
已锚定关系子集：Seen 的 S1/S2 R@8 为 65.45%/70.30%，Unseen 为 66.28%/68.02%。只保留自环会使这些子集降至52.12%/54.07%，因此消息传递确有局部作用。
有建筑子集：Seen S1/S2 为 52.74%/58.09%；Unseen 为 54.22%/52.30%，且 Unseen NE 从 41.78m 增至 44.08m，显示地标场景上的泛化问题。
无建筑子集：Seen S0/S1 R@8=3.44%，S2=21.23%；Unseen S0/S1=0.54%，S2=2.22%。此子集图只有空节点，S2 与仅自环结果相同，提升不能归因于 GAT 拓扑。主要是文本/全域位置残差学习 Seen 分布。
空地标 Prior 是全域常量，torch.topk 会在并列候选中取固定实现相关子集，造成弱基线与硬件相关的 tie 限制。额外地图中心基线的 Unseen 无地标 R@8=3.01%、NE=152.18m，优于 S2 的 2.22%、200.79m；大幅 NE 降低不足以证明真实定位改善。
S0 sigma=10/20/30/40m 的 Recall 与 NE 均相同：当前先验是 -distance/sigma，正数缩放不改变排序。这不是历史 HETT mask prior 的数值复现，也没有根据 Unseen 选择 sigma。

## 修复与验证

1. 静态转换严格过滤 Building 类型；不生成导航前缀、RGB、pose 或 teacher_future。
2. Query 去重改为 map+instruction，去重键不包含 GT；提供可重复的跨 split 清理和绑定审计脚本。
3. FrozenBERT inference tensor 在训练投影前复制，修复 backward 保存 inference tensor 的运行错误。
4. `rule_weight` 同时作用于 S1 与 S2 baseline，不再被忽略。
5. 冻结 CLS 缓存复制出 token-state 存储，修复整段特征被缓存引用的显存滞留。修复前每轮峰值最高 4.62GiB，修复后 0.60GiB，训练指标完全一致。
6. 静态几何缓存由 4096 条全局方法缓存改为 dataset 实例所有的完整缓存，避免 21k Query 在每轮重新计算和全局保留实例。
7. 每轮落盘 Checkpoint、history JSON/CSV，Loss 和吞吐按实际样本数统计，追加验证耗时和 allocated/reserved GPU 指标。
8. 保存精确 frozen_text_features；解决重新分批编码 BERT 导致一个接近并列 BF16 分数改变 Seen R@1 的复评差异。正式最终指标与逐 Query 复评的 Recall 完全一致，NE 差异 <0.0001m。
全部 38 项单元测试通过；依赖与修改代码 lint 检查通过。新增回归覆盖 Building-only 过滤、无轨迹静态转换、inference 特征 backward、CLS 缓存存储大小、rule_weight、Query 去重与目标更改不影响模型输入。原有测试还验证 pose/future/RGB 不影响静态数据和 forward 不接收目标。

## 性能瓶颈与下一步

当前 batch16、24 个 padded 图节点与 784 网格候选，模型很小，96GB 显存没有成为约束。首轮 BERT 编码和 CPU 几何预处理主导冷启动；后续 Python/DataLoader、小核与内存传输成本更明显，缓存后每轮仅约 16 秒。已有 GPU 任务使独占吞吐结论不可成立。没有证据需要提高 batch 或复杂化 GAT。
应优先建立可验证、保留同名歧义的名称别名规范，区分建筑与道路面/中心线，补齐有真实来源且对齐的道路输入；人工抽样校验关系指向目标还是其他地标，并检查建筑子集的 Unseen 退化。保留当前 5 Epoch 作为预实验，尚不据此扩展到 RGB、轨迹、Stop 或闭环性能声明。

## 产物与复现

正式运行目录：`runs/static_graph_bs16/`。`best.pt`（选中 Epoch 5）、`last.pt`（含 optimizer）、`frozen_text_features.pt`、训练日志、环境 freeze、Query 清单、两 split 逐 Query CSV、top8 距离 NPZ、诊断 JSON 均已保存。
`initial_cache_run/` 保存缓存修复前的 5 Epoch 结果，`cache_fixed_run/` 保存缓存修复后但未保存精确 BERT 特征的复评记录。最终 run 增加特征落盘后重跑同一 seed/配置。未根据 Unseen 调整模型或超参数，三次的正式训练指标一致。报告耗时仅为最终单次 5 Epoch，不把复跑总耗时误报为一次训练。
核心 CSV/JSON 与审计元数据复制到 `reports/static_geometry_20261009/` 随修复 Commit 保存；大型权重与源数据保留本地，未上传。所有源数据和 Checkpoint 的 SHA256 见 run_manifest.json。

```bash
pip install -c configs/static_geometry_rtx6000_constraints.txt -e ".[dev,foundation]"
python -m scripts.export_bounds --mapdata /home/20260922_1/Workspace/hett-landmark-multimodal-map/multiagent/mapdata.py --bounds-out data/citynav_bounds.json --ground-out data/ground_levels.json
# 每个 split 用 prepare_citynav --static-only 转换（objects、processed、trajectories、bounds 指向上述源文件）
python -m scripts.prepare_static_splits --data-dir data --output runs/static_graph_bs16/query_deduplication.json
python -m scripts.audit_static_data --data-dir data --source-dir /home/20260922_1/Workspace/DATA/citynav/extracted/data --output runs/static_graph_bs16/data_audit.json
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m graphnav.static_experiment --train data/graphnav_train_seen.jsonl --val-seen data/graphnav_val_seen.jsonl --val-unseen data/graphnav_val_unseen.jsonl --device cuda --batch-size 16 --epochs 5 --max-roads 0 --output runs/static_graph_bs16
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m scripts.diagnose_static_run --data-dir data --output runs/static_graph_bs16
```
