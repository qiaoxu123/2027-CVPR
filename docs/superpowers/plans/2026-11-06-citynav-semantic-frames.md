# CityNav semantic frames v4 Implementation Plan

**Goal:** 用语言与公开地图独立确定每条具名地标关系的参考系，冻结后才用TRAIN GT检验。
**Architecture:** v3账本全部复用；仅文本作用域规则有独立依据时扩展绑定。观察区域用可行方向范围，建筑PCA保留无向。道路增加不依赖目标的全轮廓中心线及有向场；区分常朝向、沿路方向场和不能辨认。
**Tech Stack:** 现有conda Python、spaCy、Shapely、SciPy/skimage、matplotlib；不训练模型。
**Spec:** 本轮用户完整要求，逐关系world/observer/road/intrinsic/unknown及证据、二阶段冻结、TRAIN-only。

## Global Constraints
- 不读取Val/Test，不读未来轨迹或将yaw当作者朝向；不修改导航网络。
- 参照物绑定与方向选择不得使用目标位置；GT只供离线评价。
- 区分unique、多候选、unknown；沿路方向场与常朝向分开计数。
- 道路侧与对象角度/完整轮廓谓词分开，原v3队列保留。

## Review Focus
- 语义范围跨句/代词是否错误继承；测试两条关系不同观察方向。
- 源/目的大区域能否假装唯一视点；测试宽源区域输出候选。
- 道路分支、源在中点、曲率导致不可靠单轴；冻结有向中心线，未知分支保留歧义。
- 多地标方向声明是否偷用目标GT；排除主语坐标且约束只用其他具名对象。
- 负对照保持相同关系支持；角色交换/改观察方向的反事实只验证机制，不充当真实样本准确率。

## Tasks
- [x] 1. audit.py：扫描完整TRAIN显式语义并存词汇与绑定审计；复用v3并增加保守文本绑定，输入与GT分文件。
- [x] 2. frames.py/roads.py：带精确span与原点证据的逐关系语义帧、方向范围与独立道路场；冻结预测及SHA。
- [x] 3. evaluate.py：冻结后GT、固定队列/扩展队列、词/地图/地标类别/置信分层，cluster/Wilson CI，与v1/v2/v3对照。
- [x] 4. controls.py：旋转/错绑定/角色交换/固定地图改变观察词，重复地标指令与正反关系，逐地图留出统计。
- [x] 5. render.py/REPORT.md：真实高清图、失败案例、中文结论，严格区分参考系确定性与方向检验。
- [x] 6. tests/verify.py：无GT/未来数据、范围、公开输入复算、原账本保全；实际adversarial-review；独立分支提交推送核验。

用户明确要求直接执行；当前工具模式不能切换Plan Mode，以本文件记录并直接实施，不额外请求确认。不派子代理。

执行摘要：完整TRAIN 22002/24图；保留全部v3输入，49个新增绑定。44个独立道路场、22个可判定，21符合/1冲突；22个轮廓ambiguous未删。一般观察yaw未独立唯一确定。10项规则测试及最终全量复算见verification.json。外部adversarial-review真实调用后运行时重连失败、未返回结论。提交/远端核验以最终交付的Git SHA为准。
