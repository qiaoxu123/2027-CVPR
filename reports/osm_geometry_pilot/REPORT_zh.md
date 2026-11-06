# OSM增强Geometry：数据与接入准备阶段

## 当前决策

尚不能回答OSM是否提高Top-5。真实地理配准通过的场景为0/34；A/B/C训练和比较没有执行，没有新增Checkpoint。未将缺失测量写成零收益，未用局部网页结果冒充模型指标。

## 真实数据审计

本地Revised TRAIN为21,878条，SHA256 `c24a16ddaab820d32bb6ce44a373bc2e2b8f6d88e8d08fdff9ba165f190388b2`。固定seed42配置；使用原文和同地图公开名称目录匹配，不读取目标GT来选地标：20,959条保留，919条因没有公开具名地标文本匹配排除。逐条保留/排除文件存服务器 `GraphNav/runs/osm_geometry_train_audit_20261107/{kept,excluded}.jsonl`，不覆盖原TRAIN。该匹配覆盖不代表解析正确率，也不保证有合法Polygon。后续还需几何有效性与关系作用域检查。

本地目录记录称Revised CityNav。数目符合预期，但未取得HETT Refined发布校验值，不能仅因数目相同就宣称与HETT逐字节一致。A/B/C实验须在身份核验后启动。

## 配准资源与阻塞证据

CityRefer官方README链接 https://github.com/ATR-DBI/CityRefer/blob/main/README.md ，指向Drive数据包 https://drive.google.com/drive/folders/1FToSspQ1EzO7Pb7BwYYBSl5Zqf5c04fJ 。README列出metadata/box3d格式，没有列出配准参数；本轮尝试访问Drive目录失败，服务器到raw.githubusercontent.com亦连接重置。未下载并完整检查官方原包，所以不能声称官方没有参数。

本地现有CityNav包与CityRefer官方原包不同。34图TIF的局部仿射不等于经纬度转换；独立地理验证点尚未取得，没有执行真实RANSAC拟合。不得通过目标坐标拟合，也不能仅凭同名POI当作对应控制点。

## 本轮实现与边界

- 独立`graphnav/osm_public_vectors.py`：WGS84 GeoJSON经过明确投影CRS和已验证仿射转换至局部XY；保留建筑、道路、停车场及其他场所类型、面积、长度、包含、相交和10m邻接几何。10m是数据图边定义，不是已验证语言near含义。未提供道路围合区域构建或学习式间接推理能力。
- 配准门控要求至少3独立验证点、米制最大误差不超过固定5m可视化限值；该限值不保证小目标模型使用精度，训练前需按对象尺度再次验收。修复拟合/验证点重复泄漏检查；不允许用重合点证明独立验证。
- `configs/osm_geometry/abc_pilot.json`保存A/B/C拟议口径，空间NMS预设20m、命中20m、K=1/5/10/20。B/C学习式接入未实现，配置不是训练完成证明。
- 所有代码是独立工具，原V2/V3运行时和权重未改。公开矢量模块不消费目标GT、图像或轨迹。网页仍提供局部XY可视化，未启用OSM。
- 3项CPU测试通过：拒绝未配准、拒绝验证误差过大、合成投影与包含关系正确。合成测试不属于真实配准精度结果。

## A/B/C实际状态

|组|真实训练|Top-1/NE/独立候选Recall|Checkpoint|
|---|---|---|---|
|A 原V2|本轮未执行|未测量|保留既有V2，不改写|
|B +OSM Geometry|未启动，0配准场景|未测量|无|
|C +OSM Semantics|未启动，0配准场景|未测量|无|

因此目前无法归因形状、语义或间接关系的新增收益；也不能从旧V3热图推出OSM收益。

## 下一步最小可执行入口

先恢复官方CityRefer包访问，检索原始控制点/变换；无参数则对公开建筑角点与道路交点建立跨图对应，分别在Birmingham、Cambridge拟合并保留独立控制点。随后核验HETT清单，才能在已通过场景进行A/B/C小规模公平实验。局部网页地址服务器本机 `http://127.0.0.1:8792/`；公开访问、候选RGB-D ROI及OSM图层尚未完整接入。本阶段未训练、未执行导航或Val/Test推理。
