# CityNav–OSM：34场景城市级初始化与局部相似细化

## 结论

**34/34场景实际生成了自动结构配准候选，但不能称为34个成功地理配准。独立精度通过1个、待验证21个、自动修正不可靠12个；没有场景具有最大误差≤2m的证明。** 通过者仍是已有的Birmingham block3，复用了原5个独立点，本轮没有新增其他场景的独立地理真值。

|城市|场景数|计算完成|已验证≤5m|待验证|不可靠|已证明≤2m|
|---|---:|---:|---:|---:|---:|---:|
|Birmingham|12|12|1|8|3|0|
|Cambridge|22|22|0|13|9|0|
|总计|34|34|1|21|12|0|

**城市级统一变换尚未获得跨场景成立的证据。** 它可作为候选初始化；本轮没有场景级修正同时满足预设结构收益与位移安全条件，因此保守主结果均保留城市级变换。该选择既不证明城市级模型正确，也不证明其错误。33场景仍需独立公开参照或人工重点核验，其中12场景还需要排查不稳定匹配/结构提取；不要求重画已保存控制点。

## 实际执行与公平约束

- 实验计算源码冻结于Commit `71bd7e3d`，基于`ec492c11`两城block3已发布自动矩阵作初始化。源代码SHA、城市种子、控制点快照和OSM数据SHA均保留。后续网页和可视化改动不改变此次优化数据。
- 34张真实正射PNG及高度TIF，12 Birmingham、22 Cambridge。只使用公开地图几何、环境RGB/高度和既有人工控制点；没有读取CityNav任务清单、导航目标位置或Val/Test答案，不加载或改写任何模型权重。
- 冻结Cambridge revision15（3拟合/0验证）、Birmingham revision29（10拟合/5验证）。所有独立验证点均不参与初始化、结构优化、局部候选选择或参数调整。
- 原始34场景循环完成33、一次Cambridge block3下载发生TLS瞬断。保留`first_attempt_failure.json`后仅重试该场景成功，最终34条全部计入。没有删除失败样本、改变任务分母或优化阈值。再次恢复运行确认34条均跳过已完成结果。
- 最终逐场景处理耗时合计164.66秒（下载/CPU提取/优化/画图；不包含报告、网页安装和首次失败），无训练或GPU网络推理。原始结果目录：`GraphNav/runs/osm_batch34_20261107/`；tmux执行入口`run.sh`，逐场景`batch_status.json`支持恢复。优化器源前缀变化会拒绝混合恢复。

## 自动方法、优化与限制

使用RGB植被排除、环境高度与形态学的建筑/道路代理，基于公开城市初始定位下载当前OSM矢量。自动建筑质心对应、RANSAC和soft-L1边界距离变换细化，**只允许相似变换，不增加仿射自由度**。道路骨架/交叉点目前是候选代理，不是可靠语义分割；树木、阴影、屋顶与地面Footprint差异和地图年代差异仍未解决。

主变换选择在运行前固定：局部结构代价相对下降至少10%、最大角点位移不超过15m才采用局部修正；否则保留城市级初始矩阵。该选择函数不消费验证点或其误差。局部修正风险预设为：建筑代理<3、边缘样本<100、结构代价>10或最大角点位移>30m。**这些是不可靠自动修正的风险标志，不是实际地理误差。** 不可靠12个中，11个触发大幅修正风险，1个（Birmingham block11）只有2个建筑代理。不能由这些标志断言其城市级初始变换必然错误。

局部候选角点位移中位数23.96m，最大96.20m。Cambridge block33仅14个质心候选、4个自动内点，结构代价5.21→4.01，但角点位移96.20m；因此拒绝该局部修正。重复排屋与狭长视域可能导致歧义，这是结构图的提示，尚无逐实例真值可确定每个错误对应。

## 独立精度与结构交叉检查

|场景/方案|独立点|平均m|RMSE m|P95 m|最大m|
|---|---:|---:|---:|---:|---:|
|Birmingham block3 城市级种子/最终选择|5|1.973|2.237|3.439|3.735|
|其余33场景|0|未知|未知|未知|未知|

没有将Birmingham的5点拿去验证其他场景。修复并测试了原通用工具可能将同城验证点用于其他场景评分的问题：现在按点的`map_name`严格筛选。接受某场景修正也不会将它作为其他场景已验证修正。

尝试在全部34场景进行**未参与建筑优化的道路结构检查**：RGB/高度道路骨架候选到OSM主道路的邻近距离，保留于逐场景JSON。由于没有可靠的语义/实例对应，不能视为真实地理误差，也不能据此晋级。

另外在已验证场景，将27个公开具名Polygon与类型兼容OSM几何作邻近检查；修正了CityRefer `TrafficRoad`必须映射到道路而非建筑的类别问题。27个最近几何均相交（Polygon距离0），但宽道路区域和最近对象选择会让该指标过于宽松，**不代表完美配准或27个实例都匹配正确**，不能作为新增独立验证点。没有用地标名称直接认定地理对应。

## 网页与逐场景审核

公网：http://101.43.173.9:42024/control ，沿用既有认证。网页新增34场景状态总表，显示等级、独立点数量、RMSE/P95/最大误差、≤2m状态、审核状态、导出入口。可从行按钮跳转并查看自动透明叠加及异常图。接受/拒绝按场景保存，不会覆盖人工控制点。

控制点fingerprint阻止用已修改控制点接受旧结果；仅接受其他场景预览不会使其余候选过期。人工控制点改变则所有旧结果需重跑。自动重跑使用匹配控制点版本的城市种子并沿用批次的公开结构选择策略；结果冻结后才计算本场景独立误差。原始模型和控制器不接入这些候选。

公网浏览器实测34行、跨城市切换、Cambridge block33与Birmingham block3自动叠加，零JS错误，原控制点SHA未改变。通过场景导出HTTP200，未通过场景HTTP403。截图见`*_public.png`。

## 仅验证场景的Geometry准备

仅导出Birmingham block3：319个OSM矢量要素，包括227个建筑way、77个道路way、1个停车场，以及其他场所语义；生成1595条包含/相交/10m邻接关系和240个道路图节点。它们是OSM要素计数，不是真实独立物理实例标注数量。文件`birmingham_block_3_osm_local_xy.json`使用选定矩阵转换到局部XY，保留完整与场景相交的几何及类型标签。

未验证场景禁止导出；控制点过期、结果被拒绝或导出矩阵与当前结果不一致时也会拒绝下载。OSM关系型multipolygon洞与node-only POI尚未重建，必须记录该覆盖限制。当前导出仅作研究数据准备，**没有启动A/B/C训练、导航或正式评测**。

## 复现与下一步

完整逐场景表见`scenes_zh.md`/`scenes.csv`，机器结果见`summary.json`，来源与风险计数见`provenance.json`。独立验证1/34是证据缺口，不应靠结构分数或放宽5m阈值解决。优先在不可靠的狭长/重复结构场景改进对象对应与冠层排除，并寻找不参与优化的可靠公开参照；在此之前不能将两城全部场景OSM作为已验证模型输入。

```bash
# 使用现有服务器冻结数据和缓存恢复；完成场景跳过，不混合不同优化算法
GraphNav/runs/osm_batch34_20261107/run.sh
# 生成研究导出与网页候选（不修改人工控制点）
python -m scripts.summarize_citynav_osm_batch --run <批次目录> --data <CityNav根目录> --store <私有标注目录> --report reports/citynav_osm_batch34
```

## 全部34场景
|场景|计算|质量等级|独立点|RMSE/P95/最大(m)|选择|风险标志|
|---|---|---|---:|---|---|---|
|birmingham_block_0|完成|待验证|0|未知|城市统一|—|
|birmingham_block_1|完成|待验证|0|未知|城市统一|—|
|birmingham_block_10|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|birmingham_block_11|完成|不可靠|0|未知|城市统一|insufficient_building_proxies|
|birmingham_block_12|完成|待验证|0|未知|城市统一|—|
|birmingham_block_13|完成|待验证|0|未知|城市统一|—|
|birmingham_block_3|完成|已验证≤5m|5|2.237/3.439/3.735|城市统一|—|
|birmingham_block_4|完成|待验证|0|未知|城市统一|—|
|birmingham_block_5|完成|待验证|0|未知|城市统一|—|
|birmingham_block_6|完成|待验证|0|未知|城市统一|—|
|birmingham_block_7|完成|待验证|0|未知|城市统一|—|
|birmingham_block_9|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|cambridge_block_10|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|cambridge_block_12|完成|待验证|0|未知|城市统一|—|
|cambridge_block_13|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|cambridge_block_14|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|cambridge_block_17|完成|待验证|0|未知|城市统一|—|
|cambridge_block_18|完成|待验证|0|未知|城市统一|—|
|cambridge_block_19|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|cambridge_block_2|完成|待验证|0|未知|城市统一|—|
|cambridge_block_20|完成|待验证|0|未知|城市统一|—|
|cambridge_block_21|完成|待验证|0|未知|城市统一|—|
|cambridge_block_23|完成|待验证|0|未知|城市统一|—|
|cambridge_block_25|完成|待验证|0|未知|城市统一|—|
|cambridge_block_26|完成|待验证|0|未知|城市统一|—|
|cambridge_block_28|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|cambridge_block_3|完成|待验证|0|未知|城市统一|—|
|cambridge_block_32|完成|待验证|0|未知|城市统一|—|
|cambridge_block_33|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|cambridge_block_4|完成|待验证|0|未知|城市统一|—|
|cambridge_block_6|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|cambridge_block_7|完成|不可靠|0|未知|城市统一|unstable_local_correction|
|cambridge_block_8|完成|待验证|0|未知|城市统一|—|
|cambridge_block_9|完成|不可靠|0|未知|城市统一|unstable_local_correction|

## 工程验收

19项CPU/API/真实批次一致性检查通过（5.40s）；包括跨场景验证隔离、结构分数不能授权导出、多个场景独立接受、控制点变更后拒绝旧结果、TrafficRoad类型映射，以及网页选择策略复现冻结34场景结果。保留一次测试失败记录的原因：旧测试假设局部修正可向其他场景预览传播，新实现更严格地拒绝这种传播，已更新相应验收。Starlette的一次依赖弃用警告不影响结果。

已验证导出的OSM way保留version/timestamp；原始公共OSM XML压缩快照`birmingham_block_3_source.osm.gz`一并提供（ODbL，© OpenStreetMap contributors），其SHA与源记录一致。控制点及其他场景完整数据留服务器，不提交凭据。

最终验收：19项CPU/API/真实34场景策略一致性检查通过（5.40s）；包括验证点场景隔离、多个场景接受不互相污染、点集变化拒绝旧候选、未验证导出拒绝、TrafficRoad类型映射。公网实际点击Cambridge自动重跑确认使用冻结城市种子和相同选择规则；再次浏览34行并切换两个城市、加载自动透明图和下载已验证矢量，零JavaScript错误，原人工点集/文件SHA未变。证据见`public_auto_rerun.json`和`public_browser.json`。
