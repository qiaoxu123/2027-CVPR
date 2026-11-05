# Polygon显式先验与轻量关系训练（TRAIN）

## 结论范围
仅TRAIN、FP32、seed42；不改普通推理，不运行Val/Test或闭环。方向地理配准未完成，所有地理方向弃权，不宣称方向理解完成。

## 数据与训练
{
  "seed": 42,
  "epochs": 6,
  "precision": "FP32, TF32 off",
  "trainable_parameters": 85377,
  "fit_counterfactual_counts": {
    "inside": 388,
    "outside": 388,
    "near": 412
  },
  "dev_counterfactual_counts": {
    "inside": 49,
    "outside": 49,
    "near": 55
  },
  "fit_original_queries": 397,
  "natural_queries": 573,
  "fit_groups": 209,
  "dev_groups": 23,
  "group_overlap": 0,
  "cache_seconds": 31.322144540026784,
  "train_seconds": 13.431768885813653,
  "peak_GPU_GiB": 0.5219635963439941,
  "max_gradient_norm": 10.693325996398926,
  "source_checkpoint": "/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/unified_direct_target_s42_20261105/training_aligned/best.pt",
  "source_sha256": "aa359894a172c38b024aa5ef4e89caeecc8a496d723320cea14862f0eaaf61e9",
  "checkpoint_sha256": "9b63dfe6316cb9fc196a90bdcf78de3c1efd5539bce3cdaacb5635be19528dcb",
  "validation_used_for_selection": false,
  "epochs_and_mixture_fixed_before_validation": true,
  "near_radius_m": 30.0,
  "region_margin": 0.2,
  "preservation_KL_weight": 0.2,
  "mixture": 0.5,
  "geography_verified": false,
  "directions_abstained": true,
  "Val_Test_accessed": false,
  "normal_runtime_changed": false
}

## 目标定位（相对冻结Geometry）
|分组|方法|N|R20|NE(m)|GT20m质量|纠正|破坏|净收益|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|dev_all|Geometry|80|46.25%|38.4238|0.24654|0|0|+0|
|dev_all|Explicit|80|33.75%|46.4951|0.16064|2|12|-10|
|dev_all|Relation|80|41.25%|41.8017|0.22581|3|7|-4|
|dev_all|Fusion|80|38.75%|43.8290|0.19323|3|9|-6|
|dev_supported|Geometry|46|50.00%|33.8687|0.28521|0|0|+0|
|dev_supported|Explicit|46|28.26%|47.9058|0.13582|2|12|-10|
|dev_supported|Relation|46|41.30%|39.7433|0.24916|3|7|-4|
|dev_supported|Fusion|46|36.96%|43.2690|0.19249|3|9|-6|
|dev_inside|Geometry|16|50.00%|20.6872|0.24450|0|0|+0|
|dev_inside|Explicit|16|18.75%|33.5328|0.17833|0|5|-5|
|dev_inside|Relation|16|31.25%|34.8021|0.19841|1|4|-3|
|dev_inside|Fusion|16|18.75%|36.9341|0.18837|1|6|-5|
|dev_near|Geometry|30|50.00%|40.8988|0.30692|0|0|+0|
|dev_near|Explicit|30|33.33%|55.5713|0.11314|2|7|-5|
|dev_near|Relation|30|46.67%|42.3786|0.27623|2|3|-1|
|dev_near|Fusion|30|46.67%|46.6476|0.19469|2|3|-1|

## 反事实区域响应（独立地图组TRAIN dev）
|方法|IO共同对数|联合响应率|inside质量差|质心位移m|
|---|---:|---:|---:|---:|
|Geometry|49|44.90%|-0.00280|3.878|
|Explicit|49|100.00%|0.60596|142.216|
|Relation|49|100.00%|0.69973|120.555|
|Fusion|49|100.00%|0.65284|129.191|

inside/outside是拓扑变换，没有单一正确位移方向；质心移动大小不作为方向正确率。near与outside有重叠，不能人为当成反义词。

## 决策与限制
Fusion相对Geometry纠正3、破坏9，净收益-6；Relation净收益-4。
不自动开展56条Val闭环。区域响应改善并不等于真实目标识别，需先检查目标定位净收益、NE及跨地标不确定性。
dev按地图与公开地标ID隔离，但此前已用于诊断，非新的未触碰确认组；Geometry历史预训练也可能用过TRAIN，不能称全模型未见泛化。
反事实没有目标GT；只使用公开Polygon构造区域标签。inside/outside/near的区域监督是几何弱监督，不是人工像素标注。真实目标GT仅用于原始fit指令训练与dev事后评价。
原始权重保留不变；只训练现有Relation组件，不训练RGB、PGVB、Geometry、Stop或StableGoal。
未进行地理方向配准：24地图TIF缺少CRS；公开对象表只有本地XY，官方rasterize.py未提供局部到地理的旋转与原点。需要官方地理元数据或至少两个可靠非GT地理控制点。不得根据目标GT翻转坐标。

### 地标组bootstrap R20差95%区间
{
  "Geometry": [
    0.0,
    0.0
  ],
  "Explicit": [
    -0.23809523809523808,
    -0.0405270270270271
  ],
  "Relation": [
    -0.125,
    0.0
  ],
  "Fusion": [
    -0.15294117647058825,
    -0.011232833957553076
  ]
}

## 可视化
图中青色为完整公开Polygon、红叉为目标GT，仅用于离线评分展示；各方法共享世界坐标与28×28网格。

## 事后冲突审查与真实参数更新

```json
{
  "changed_parameter_maxabs": {
    "direct_target.word.weight": 0.02477813884615898,
    "direct_target.word.bias": 0.018085192888975143,
    "direct_target.predicate.weight": 0.04888415336608887,
    "direct_target.relative.0.weight": 0.052168428897857666,
    "direct_target.relative.0.bias": 0.03073469176888466,
    "direct_target.relative.2.weight": 0.043466389179229736,
    "direct_target.relative.2.bias": 0.01648583449423313,
    "direct_target.relation_score.0.weight": 0.04628733545541763,
    "direct_target.relation_score.0.bias": 0.01876344345510006,
    "direct_target.relation_score.2.weight": 0.04352150112390518,
    "direct_target.relation_score.2.bias": 0.012058310210704803
  },
  "frozen_modules_unchanged": true,
  "source_sha256": "aa359894a172c38b024aa5ef4e89caeecc8a496d723320cea14862f0eaaf61e9",
  "checkpoint_SHA256": "9b63dfe6316cb9fc196a90bdcf78de3c1efd5539bce3cdaacb5635be19528dcb",
  "relation_GT_compatibility": {
    "fit": {
      "inside": {
        "n": 9,
        "GT_in_relation": 1,
        "empty_grid_region": 0,
        "median_polygon_distance_m": 18.211427981700275
      },
      "near": {
        "n": 388,
        "GT_in_relation": 365,
        "empty_grid_region": 0,
        "median_polygon_distance_m": 8.598112356002133
      }
    },
    "dev": {
      "inside": {
        "n": 16,
        "GT_in_relation": 0,
        "empty_grid_region": 1,
        "median_polygon_distance_m": 7.72626951115242
      },
      "near": {
        "n": 30,
        "GT_in_relation": 23,
        "empty_grid_region": 0,
        "median_polygon_distance_m": 10.622806479686957
      }
    }
  },
  "near_common_n": 49,
  "near_response": {
    "Geometry": {
      "rate_vs_outside": 0.40816326530612246,
      "mean_mass_delta_vs_outside": -0.013613481302650608,
      "mean_mass_delta_vs_inside": -0.008610278063890885
    },
    "Explicit": {
      "rate_vs_outside": 1.0,
      "mean_mass_delta_vs_outside": 0.5965388639620981,
      "mean_mass_delta_vs_inside": -0.3221936712459642
    },
    "Relation": {
      "rate_vs_outside": 1.0,
      "mean_mass_delta_vs_outside": 0.4984532341602192,
      "mean_mass_delta_vs_inside": -0.12200861074486558
    },
    "Fusion": {
      "rate_vs_outside": 1.0,
      "mean_mass_delta_vs_outside": 0.5474960589591338,
      "mean_mass_delta_vs_inside": -0.22210111788340978
    }
  },
  "caution": "near overlaps inside/outside; comparison is distribution response, not antonym correctness"
}
```

几何区域约束不能在同一区域内区分多个目标。均匀区域CE存在抹平目标尖峰的风险，原始语言的near并无固定30m定义；这些是本轮验证的假设，不能用dev结果重新选半径。下一轮优先将区域关系作为排序约束并保留原Geometry区域内部相对概率，而不是扩大网络或把均匀区域当真实目标标签。此建议尚未验证。

### 地理配准外部证据
原始PLY头只有XYZ/RGB/class，无CRS。上游公开仓库[坐标问题#63](https://github.com/QingyongHu/SensatUrban/issues/63)及[参考原点#56](https://github.com/QingyongHu/SensatUrban/issues/56)尚未给出可用变换；不能据此反推轴向。方向实验仍被独立元数据阻塞。

## 复现与保存
训练6轮450个有效更新，Loss 12.18186→7.91844。缓存首次生成31.32秒，缓存后的训练13.43秒，峰值0.522GiB（PyTorch allocated，非整卡占用）。仅85377参数可训练；最大裁剪前梯度10.6933，裁剪阈值5。

Checkpoint仅保存Relation Head与优化器，不是完整UnifiedPGVB部署包；通过原模型加载`last.pt["head"]`到`model.direct_target`恢复，原始Checkpoint SHA未变。完整缓存/权重保存在服务器，不提交大文件。路径和SHA见manifest/provenance。

反事实样本原始数量388/388/412，通过逆频率损失权重平衡三类；不宣称原始样本数相等。TRAIN原始目标监督397条。独立地图dev共同IO49组；不是49个独立地标，聚合时保持对象分组。

### inside语义审查
25条原始inside任务中只有1条GT实际位于绑定Polygon内部（fit1/9，dev0/16）。例如`in Chesterwood Gardens`指街区住宅、`in the Graduate Cambridge building`描述停车车辆、`within Queens College Library quadrangle`指庭院。这些暴露了公开对象Footprint与语言区域作用域不一致；不能把语言in一律编译为严格Polygon内部。该统计仅用于事后诊断，不筛选dev或重训。

本轮未使用RGB，因此不能就视觉实例识别能力作结论；也不能从区域响应100%推断真实任务关系理解100%。当前选择保留负结果，停止在TRAIN阶段；下一轮优先审计并拆分街区/庭院/停车区域与物体Footprint关系，之后再用独立TRAIN分组验证保留Geometry区域内排序的约束损失。

## 测试
348项受版本管理CPU测试通过，1项上游rasterio弃用警告；包含5项新增Polygon/区域损失测试。Checkpoint重载与优化器450步状态完整。HTML及其全部本地图片/报告链接通过HTTP200检查；未宣称交互浏览器测试。此前未提交的instance-memory实验保持不变，未计入本轮测试口径。

独立adversarial-review已实际调用，但发生多次连接重试，未取得结构化审查结果；不计为审查通过，原始输出保留在adversarial_review_attempt.txt。
