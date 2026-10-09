# GraphNav 技术设计：图主导的独立 CityNav 语言导航

## 1. 目标

替换旧的 HETT Transformer / 两阶段网络，使用可解释的拓扑—几何联合推理。
**独立实现，不通过复用 HETT 网络假装换名**。仅保留 CityNav 数据接口与测评任务定义。

核心困难不是获取更多视觉 embedding，而是正确地绑定具名地标和轮廓，并使“东侧 / 附近 / 沿道路 / 两者之间”等语义对应候选目标的真实几何结构，同时在未知处保留信念。

## 2. 模块与张量

| 模块 | 输入 | 输出 | 训练 |
|---|---|---|---|
| Named Landmark Binding | Map object ID, name, ordered contour | ID-Name-Contour pair | 确定性，不训练 |
| Frozen BERT | instruction + landmark names | 768-d sentence/name embeddings | 冻结，可缓存 |
| Contour Encoder | P=16 polygon vertices, center/size/area/perimeter | 128-d node | 训练 |
| Static/Dynamic GATv2 | UAV + M<=16 landmark nodes; [dx,dy,range,bearing,yaw] edges | 128-d relational nodes | 训练 |
| Dense Goal Field | 28x28 cells + instruction + landmark graph + signed polygon distance | [784] goal logits | 训练 |
| Proposal GATv2 | Top-20 goal + landmark + UAV | top-K candidate graph states | 训练 |
| RGB Memory | Observed RGB + camera footprint | causal local RGB mosaic | 不训练 |
| Frozen SigLIP2 | only valid candidate image crops | 768-d visual embeddings | 冻结 |
| Visual Residual | image embedding + proposal state + visibility | gated candidate score delta | 训练 |
| Trajectory + Stop | candidate/UAV/graph/language | K x 3 x 8 x XY + stop logits | 训练 |

各类图节点使用 Node Type Embedding；边用带**有符号东西/南北位移**的 8 维几何，候选↔地标边最后一维使用轮廓的**有符号边界距离**。这与只计算地标中心的方案不同，也没有引入需要复杂安装的 PyG。

### 为什么对全部网格先评分，再做 Top-K 图？

不可以仅把 Top-K 视作真实候选集：错误初始预测会导致 GT 永远无法出现。Dense Goal Field 对 28x28 全格加软高斯标签监督，图候选层只在 Top-20 上做精化和路线选择。若 Top-K Recall 不足，应该先优化 Dense Goal Field，而不是仅调轨迹或 Stop。

### Graph Attention

对方向边 (i,j)，GATv2 权重依赖有向相对几何：

\`\`\`text
a_ij = softmax_j( score(LeakyReLU(Wq hi + Wk hj + We e_ij)) )
h'_i = LayerNorm(hi + Wo sum_j a_ij (Wv hj + Wve e_ij))
\`\`\`

注意固定世界方位 north/east 词必须保留世界坐标下有符号 dx/dy；无法只输入旋转不变的两点欧氏距离。

## 3. 视觉门控与双时序

- 地标 ID、轮廓、BERT embeddings：静态；每 Episode 生成、缓存。
- UAV pose：每步更新；图关系重新计算。
- ObservedRGBMemory：仅用本步**实际观测像素**更新历史。当前 RGB 不相关也照常存，未来可能相关。
- Candidate crop：由 Top-K 的世界坐标索引历史观测 Mosaic，输出 RGB、有效 Mask、覆盖率、中心覆盖率。
- SigLIP2：有效候选才运行；已观测但特征完全不变时复用 LRU。
- Candidate score：仅有足够有效观测时添加视觉残差。未观测 = 0 视觉增量，不等于负样本。
- Visual refinement：只运行轻量候选头，**不重复计算**全图 Dense Goal Field。

注意：一张候选裁剪图即使含少量已观测背景，也不等于候选实体本身可见；需同时检查中心覆盖率。后续需在具象建筑轮廓上做更严格 ROI/可见性消融。

## 4. 训练目标与 teacher 隔离

- \`L_goal\`：soft spatial Gaussian cross-entropy，监督 784 网格，sigma=0.9 cell。
- \`L_rank\`：候选中软目标分布，训练提案排序；不能补偿 GT 丢失 Top-K。
- \`L_path\`：最近目标候选的 best-of-modes ADE，监督**实际人类剩余路径**，不生成 GT 直线作为 teacher。
- \`L_stop\`：使用真实目标距离制作二元训练标签，但闭环必须独立验证 False-Stop。

所有 GT、teacher_future、目标距离只出现在 \`graphnav.losses\` 或监督数据中，不在网络 forward 参数中。

初始基线仅支持离线 teacher trajectory 训练。获得有意义的视觉门控训练，还需给训练样本附加实际/按仿真足迹生成的 RGB 观测及目标候选区域，并采用在线环境 rollout；不能假称已有训练结果。

## 5. 决策速度

- 静态 text embedding 缓存；BERT 不每动作重复执行。
- GAT 图不包含 784 个网格，只有 1 UAV + 最多16地标，或最多37 nodes。
- Dense Goal Field 使用按64网格分块的 polygon segment distance；不存在全图全连接 GNN。
- 冻结 SigLIP2 只对已观测候选批量运行。
- BF16/DDP 训练入口；请本地测实际 it/s、GPU 显存、RGB 编码耗时。
- 不对训练速度、参数量和 SR 做未实测的数字承诺。

## 6. 路线与止动

模型为每个目标输出 3 条不同潜在模式、每条8个点，全部以对应候选为终点。每个环境 step **只执行第一 waypoint**，下一次新观测后重规划。

当前的 trajectory mode 具备多样性输出参数，但不包含显式的障碍物/动力学可行性约束；实际模拟应增加可飞空间碰撞检查、进度监测和多样性约束。默认不依据 stop logits 自动下达停止动作，以避免复现旧版早停失败。
