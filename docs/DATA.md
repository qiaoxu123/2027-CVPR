# CityNav 数据适配

GraphNav 使用**独立 JSONL**，避免运行时导入 HETT 代码、加载其网络或 checkpoint。

## 必要数据

来自 CityRefer 和 CityFlight/CityNav 的 JSON：

- \`cityrefer/objects.json\`：地标的 map, object ID, name, contour
- \`cityrefer/processed_descriptions.json\`：每个描述提及的具名地标列表
- \`processed_citynav/citynav_train_seenall.json\` 等：人工飞行轨迹、instruction索引与目标监督
- \`citynav_bounds.json\`：已知 map block 的固定物理 XY 范围
- \`ground_levels.json\`：可选，但建议提供，用于从历史 UAV poses 创建**因果探索覆盖 Mask**

## 转换

\`\`\`bash
python scripts/export_bounds.py \
 --mapdata /path/to/HETT/multiagent/mapdata.py \
 --bounds-out data/citynav_bounds.json --ground-out data/ground_levels.json

python scripts/prepare_citynav.py \
 --objects data/cityrefer/objects.json \
 --processed data/cityrefer/processed_descriptions.json \
 --trajectories data/processed_citynav/citynav_train_seenall.json \
 --bounds data/citynav_bounds.json --ground-levels data/ground_levels.json \
 --output data/graphnav_train_seen.jsonl
\`\`\`

Val seen/unseen 用各自的 \`citynav_val_seenall.json\`、
\`citynav_val_unseenall.json\`，不得将验证地图的统计泄漏进训练。

每个记录由指定 human 轨迹前缀创建。地标选择只看指令中引用的名字和公有 map data，不使用 GT target 消歧。若同名对应多个轮廓，保留多个 ID（上限 M=16），不能通过 GT 最近地标选择。

JSONL 关键字段（坐标统一 [0,1]、x east/y north）：

\`\`\`json
{
  "episode_id": "cambridge_block_3:object_id:desc_id:step",
  "instruction": "Find the white building near the church",
  "landmarks": [
    {"id": "21", "name": "St Mary's Church",
     "contour": [[0.21, 0.30], [0.23, 0.30], [0.23, 0.32]]}
  ],
  "uav_pose": [0.10, 0.12, 0.0, 1.0],
  "spatial_maps": "[2,28,28] numeric nested array",
  "target_xy": [0.25, 0.33],
  "teacher_future": "[8,2] numeric nested array",
  "physical_map_sizes": [400.0, 400.0]
}
\`\`\`

\`target_xy\` 和 \`teacher_future\` **仅**作为 Loss 标注。

### RGB 数据的重要限制

原始数据含全局正射图，但即使图像文件存在，也不能直接给 GraphNav 读取未探索部分。
\`graphnav.memory.ObservedRGBMemory.observe()\` 只接受当前观测及其地理范围；
如需训练 SigLIP2 残差，必须在模拟 rollout 中把真实已观测 RGB 投影到历史图上，然后获得候选图像与有效 Mask。

默认转换只生成因果视野**二值 Mask**，初始训练为 **geometry/language first**。

可选提供 \`--rgbd-dir data/rgbd --observations-out data/observed_train\` 及 \`--ground-levels\`。转换器只为每个训练步骤保存到该时刻为止的各张俯视 RGB 图和对应 footprint，JSONL 追加 \`world_bounds\` 与 \`rgb_observations\`，不会将全域正射图传入模型。

使用 \`python -m graphnav.train --data <file> --use-vision\` 后，训练阶段将：
ObservedRGBMemory 累积过去观测 → 取当前网络 Top-K 的真正可见候选 → 冻结 SigLIP2 → 可见性门控的图候选评分残差。未知区域只产生零视觉增量，不作为负样本。

这仍是**离线轨迹监督**；真实闭环需要进一步接入 CityNav simulator 的 reset/step/碰撞和动作反馈。

### 坐标与地图方向

- 输入世界坐标为米，x 向东、y 向北。
- 地图图像行数向下增加，转换时必须反转 y。
- CityNav 使用相对地面高度近似正方形 nadir 视野；真实平台应替换相机 footprint 计算。
- 图片与世界足迹四角一一对应：TL,TR,BR,BL；禁止将四角随意排序。
