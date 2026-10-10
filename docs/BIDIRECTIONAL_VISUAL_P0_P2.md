# GraphNav P0–P2：双向几何预训练与沿途视觉地标

**基线**：\`validation/full-token-rgb-local-20261009\` @ \`8a75d9e\`。独立实验分支，不修改原验证 Commit、CSV 或模型。当前真实 Revised CityNav 结果：A3 Val Unseen R@8/20m **50.88%**，A4 **48.00%**，A5 **48.11%**。A5 相比 A4 +0.11pp，远未证明稳定。仅 32.81% 的 Unseen Query 有有效初始 RGB；不能把终点 RGB 小范围重排序误当成沿途导航视觉理解。

## 0. 明确任务边界

本分支交付**三个独立、可验证的研究工具**：

- **P0 Map↔RGB QA**：将 CityRefer WORLD 米制多边形投影到真实已渲染的 CityFlight 观测帧，导出每帧完整可见的具名 ROI、可见面积和人工检查叠加图。**投影不等于视觉语义分割真值，也不等于可靠的实体 re-ID**。
- **P1 可逆关系预训练**：源地图中两个可靠具名地标 A/B 生成成对的“B 在 A 东侧 → A 在 B 西侧”文本；目标 ID/名称/多边形不得进入该方向 forward。学习全局坐标目标分布+正反相对位移一致性，另提供相对坐标特征控制。
- **P2 沿途视觉预训练**：从独立时间步的**实际观测帧**投影 ROI，用*同一真实地图地标 ID 的两次观测*构造正例，同城市同类别其他地标作为负例。冻结 SigLIP2 Large/384，保留 ViT patch tokens；训练图像 patch attention、实例对比和类名 BERT 文本对齐。注意 P2 使用的 polygon ROI 是**需要核查的弱监督区域**，不是自动验证过的真值。

**尚未实现**：P2 权重与 A3 的动态视觉信念融合、由策略自身访问的 RGB memory、Waypoint、Stop、闭环 SR/SPL。本轮不能声明这些功能已经完成。不要依据 P1 合成 R@k 或 P2 同物体跨帧检索指标宣称 CityNav 目标定位已提升。

## 1. 输入与防泄漏协议

必须使用**同一 Revised CityNav 修订版**的 objects、processed、splits、米制边界、ground levels、RGB PNG/GeoTIFF。以已验证报告中的 SHA256 作为校验基准。

P0/P2 从 NON-static CityNav 转换器读取**每条观测当前时刻之前**的 RGB，绝不把目标终点图像或未来人类轨迹作为模型当前视觉。地标 ID 只在 P2 数据构造中选正负对应，ROI 模型 forward 只有真实像素 Patch；P1 目标 ID、绝对坐标只作为标签，forward 输入只包含单一 anchor 多边形和语言。不可把 generated reverse records 跨 train/val maps 混用。

Camera footprint TL/TR/BR/BL→像素四角的单应性使用 CityNav 正射相机模型，与原 cropclient/renderer 同一几何约定；不声称适用于未标定的真实透视相机。**人工检查叠加轮廓是否落在正确屋顶/道路/桥梁**，有系统误差先修正 TIF bounds/世界坐标/Yaw 约定。

## 2. P0 — 构建真实沿途帧与检查投影

完整资产示例中 \`$DATA\`、\`$RGBD\`、\`$BOUNDS\`、\`$GROUND\` 按本机实际路径填写，绝不从提示词猜文件名。以下只示范 Train；同样分别处理 Val Unseen。

\`\`\`bash
git fetch origin
git switch 2027-CVPR/bidirectional-landmark-visual-pretrain
pip install -e '.[dev,foundation]'
pip install rasterio

mkdir -p data/revised_dynamic
python -m scripts.prepare_citynav \
  --objects "$DATA/cityrefer/objects.json" \
  --processed "$DATA/cityrefer/processed_descriptions.json" \
  --trajectories "$DATA/citynav_train_seen.json" \
  --bounds "$BOUNDS" --ground-levels "$GROUND" \
  --rgbd-dir "$RGBD" \
  --observations-out data/revised_dynamic/frames/train_seen \
  --output data/revised_dynamic/graphnav_train_seen.jsonl \
  --dataset-version revised --match-mode fuzzy \
  --max-steps 64 --stride 8 \
  --render-observed-only --render-image-size 384
\`\`\`

区别于以前的 A5：这里会生成**人类轨迹中的多个采样观察时刻**（仅 teacher path 观察回放诊断，不是模型闭环），而不是只有 step=0。渲染采用**真实的 384px 视图采样**，不是将 224px 图片放大。每行的 \`rgb_observations\` 严格按 \`frame.step <= observation_cutoff_step\` 过滤。可用 \`--max-steps 16\` 先烟雾验证，之后全轨迹根据硬盘容量调整。

\`\`\`bash
python -m scripts.prepare_landmark_views \
  --records data/revised_dynamic/graphnav_train_seen.jsonl \
  --objects "$DATA/cityrefer/objects.json" \
  --out-dir runs/p0_alignment/train_seen \
  --save-rois --roi-size 384 \
  --min-visible 0.8 --min-pixels 80 \
  --max-per-frame 16 --frame-sample-ratio 0.2 \
  --overlay-frames 80
\`\`\`

Val Unseen 同样生成自己的 \`graphnav_val_unseen.jsonl\` 和 \`runs/p0_alignment/val_unseen\`，不能复用训练的 RGB 或混用地图。

检查：
- \`alignment_summary.json\`：有效帧/ROI 数、类型分布、几何视野覆盖比例。
- \`overlays/*.png\`：人工核验屋顶、道路等轮廓是否落在正确视觉区域；特别检查图像上下翻转、旋转、TIF offset 与建筑高差。
- \`landmark_rois.jsonl\`：原始 source map+object_id、step、实际帧路径、像素 contour、视觉几何可见面积及 ROI 文件。**无 GT 目标、无 teacher_future**。
- **不可**将投影轮廓与自己生成的 Mask 算 IoU 当“配准准确率”；如需真正 IoU，另外准备可靠人工分割/独立图像特征标注。

如果不同步、缺少真实观测、ROI 不可靠，**P2 暂停**，先修投影和生成协议。

## 3. P1 — 可靠可逆方向关系（无伪造返程语言）

\`\`\`bash
mkdir -p data/p1_pairs
python -m scripts.build_bidirectional_pairs \
  --objects "$DATA/cityrefer/objects.json" --bounds "$BOUNDS" \
  --source-split data/revised_rgb/graphnav_train_seen.jsonl \
  --output data/p1_pairs/train.jsonl \
  --max-pairs-per-map 800 --shift-augmentations 2

python -m scripts.build_bidirectional_pairs \
  --objects "$DATA/cityrefer/objects.json" --bounds "$BOUNDS" \
  --source-split data/revised_rgb/graphnav_val_unseen.jsonl \
  --output data/p1_pairs/unseen.jsonl \
  --max-pairs-per-map 800 --shift-augmentations 0

python -m graphnav.bidirectional_train \
  --train data/p1_pairs/train.jsonl \
  --unseen data/p1_pairs/unseen.jsonl \
  --output runs/p1_inverse_geom \
  --device cuda --epochs 5 --batch-size 32 \
  --hidden 256 --heads 8
\`\`\`

每个 \`batch-size 32\` 是 **32 对、64 条 forward**，不能与先前 A4 的 128 条 Query 混为一谈。请按真实 GPU samples/s 和主机吞吐确认最佳组合。

成对的 relation 只覆盖几何能验证的 east/west/north/south，允许真实距离近似量词；不机械反转 \`between\`、\`along\`、路径中的 \`left/right\`，也不把反向航迹伪装成有人类语言标注的专家轨迹。增广将地标和目标一起平移，世界相对位移不变；所有正反派生记录 stay in SAME split。没有 GT 坐标辅助挑选推理 anchor。另需报告构造成功率、类别、地图以及“同场景模糊多解”比例（距离约束不一定能保证语言唯一）。

**如何判断 P1 真的对正向定位有益？** 使用相同的真实 Revised CityNav 数据和训练配置，分别运行 A3 从头训练和 P1 初始化 A3。两个实验必须一样的 hidden=256、heads=8、\`--relative-position\`、seed、Epoch、batch size；只改变是否加入 \`--init-bidirectional\`。

\`\`\`bash
# Control: relative-coordinates A3 scratch
python -m graphnav.full_experiment \
  --train data/revised_rgb/graphnav_train_seen.jsonl \
  --val-seen data/revised_rgb/graphnav_val_seen.jsonl \
  --val-unseen data/revised_rgb/graphnav_val_unseen.jsonl \
  --dataset-version revised --match-mode fuzzy \
  --variants geometry --relative-position \
  --hidden 256 --heads 8 --epochs 5 --batch-size 64 \
  --workers 4 --device cuda --output runs/p1_a3_scratch

# P1-initialized A3: same protocol; different initialization only
python -m graphnav.full_experiment \
  --train data/revised_rgb/graphnav_train_seen.jsonl \
  --val-seen data/revised_rgb/graphnav_val_seen.jsonl \
  --val-unseen data/revised_rgb/graphnav_val_unseen.jsonl \
  --dataset-version revised --match-mode fuzzy \
  --variants geometry --relative-position \
  --init-bidirectional runs/p1_inverse_geom/pretrain_best.pt \
  --hidden 256 --heads 8 --epochs 5 --batch-size 64 \
  --workers 4 --device cuda --output runs/p1_a3_initialized
\`\`\`

Check selected Val Seen R@8, Unseen R@1/4/8, NE, paired significance, map-wise results; verify translation/word-direction counterfactuals. P1 adds training data and compute: compare **total budget** with a sufficiently trained scratch control before attributing gains solely to reverse relations.

## 4. P2 — Patch-level landmark matching while target is FAR away

Prerequisites: P0 train and unseen ROI JSONL, \`--save-rois\`, sufficient \`(map_name, object_id)\` groups each with ≥2 different observed frames, and manually inspected overlays.

\`\`\`bash
python -m graphnav.landmark_visual_train \
  --train runs/p0_alignment/train_seen/landmark_rois.jsonl \
  --unseen runs/p0_alignment/val_unseen/landmark_rois.jsonl \
  --acknowledge-geometry-pseudo-labels \
  --image-backbone google/siglip2-large-patch16-384 \
  --output runs/p2_visual_landmarks \
  --device cuda --batch-size 16 --image-microbatch 8 \
  --hidden 256 --heads 8 --epochs 5
\`\`\`

网络保留 \`vision_model.last_hidden_state\` 的 **patch tokens**，而非仅用 \`get_image_features\` 池化向量；用 trainable attention 汇聚，训练两个不同时间的 **同一 source ID** ROI 相似度，及图像↔类型词特征对齐。图像前向**没有 entity ID**；它只在 InfoNCE 的配对标签和检索统计中出现。默认相同城市、相同类型的不同 source ID 被编成 hard-negative batches。

R@1、R@5 是 **landmark visual cross-view retrieval**（一个 view 查询 vs 另一不同帧的 gallery），**不是 CityNav target-location R@k/20m**。Holdout Unseen 地图只评估一次、不用于选 Checkpoint。

冻结 SigLIP2 Large 的微批大小默认 8；即使 GPU 可处理旧 A4 的 batch128，也**不能直接**把两次高分辨率 Patch 前向的 batch128 当成显存等价。先查看样本量/吞吐和 PyTorch peak GiB 后再扩展。当前 Python/HF 模型权重的 \`vision_model\` Patch API **尚需本地 GPU+Transformer 真模型验证**，CI 的 CPU 测试使用合成 patch tensor；不保证不同 Transformers 版本上直接可用。

## 5. 可复现检查与继续/停止条件

1. P0：原始帧 + projected overlay 人工可信，报告可见 ROI 的地图/类别分布，不把轮廓本身的数学闭合当作真实语义对齐准确率。
2. P1：与**重新训练且同配置** A3 scratch 比，Val Unseen R@8/20m、NE 和方向反事实指标是否改善；避免只看训练合成 pair loss。
3. P2：Unseen 地图上真正不同帧同地标检索率、同类困难负样本和人工抽样 ROI 错配率是否可信。
4. 若任一模块失败，优先修数据监督、标签置信度、物理坐标或输入泄漏，不增加 GAT 层数。
5. P1、P2 分别独立通过才进入 **P3：动态可见区域的 Visual Belief Update → Waypoint / Stop**。本分支**没有**训练 P3/P4，也不使用离线教师轨迹的未来帧做闭环模拟。

运行 \`pytest -q\`、\`python -m scripts.prepare_landmark_views --help\`、\`python -m graphnav.bidirectional_train --help\`、\`python -m graphnav.landmark_visual_train --help\`。务必记录每一步执行命令、SHA256、Epoch 时间/GPU memory、对照指标，不合并 \`main\`。
