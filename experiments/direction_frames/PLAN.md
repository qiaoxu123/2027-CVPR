# CityNav TRAIN 空间词参考系独立实验计划

**Goal:** 用完整原始 TRAIN 和原始 CityRefer 对象，检验方向词是否具有统一 XY 约定；不训练、不改网络、不读取 Val/Test。
**Architecture:** 文本层保留每次出现及关系作用域；独立几何层通过 GT 离线验证，绝不用 GT 消歧具名地标或选择参考方向；统计层同时报告覆盖率、排除原因、Polygon 检验及假设限制。
**Tech Stack:** Python (existing graphnav conda), numpy/pandas/scipy/shapely/rasterio/matplotlib。
**Spec:** 用户本会话 1–7 项要求。

- [x] 审计 TRAIN 完整性、原始注释匹配、pose、地图 affine、地理方向证据；记录 SHA256。
- [x] 创建完整方向词 occurrence 清单（含地名/多义词/非关系）及直接/间接/未知作用域；仅唯一名称匹配、保守句法绑定进入几何统计。输出机器可审计证据及样本审读。
- [x] 用 position 与完整 Polygon 计算 ΔXY/角度/相交/边界投影，预设 1m 容差并做 0.5/2m 敏感性；不以质心掩盖轮廓冲突。
- [x] 比较 A 固定 XY 最佳轴（描述性）及分组留出拟合；B 初始 yaw；C 起点到参照物视线；D 可信自身朝向（无字段则不可评估）。输出 Wilson 及按目标 cluster bootstrap 95% CI。
- [x] 绘制真实直方图、极坐标、分组、对照及符合/冲突/未知案例。中文报告回答问题并明确样本选择与描述者参考系的不可识别性。
- [x] 运行句法/几何/统计测试和全量工件完整性检查；独立分支 commit + push。

## Review focus

1. 目标—停车场—具名地标的间接作用域不可扁平化。
2. 同名对象不可用 GT 最近距离选取；复合词/专名/强调词不可误当关系。
3. 同目标多描述及重复英文不能作为独立证据夸大 CI。
4. A 学到的词轴不能与 B/C 的预设语义同样当独立假设证明；提供留出/共同支持集。
5. 图像上方不是已校准地理北；Polygon 主轴不是有向 heading。

外部对抗审查已实际调用但服务重连失败，详见 review/STATUS.md；本地验证成功不代表外部审查通过。
