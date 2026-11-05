# 完整地标RGB与多尺度PGVB：TRAIN特权输入实验

状态：数据与接口预检完成，全量缓存续跑中；冻结Dev四组预测已封存；尚无目标定位增益结论。

基于PR18 `ef19a7fa941d0e4e3e12a91c31e60d2b39417243`，分支`experiment/complete-landmark-rgb-training-20261105`。不修改原导航入口、模型权重、Stop/Goal阈值，不访问Val/Test。本实验没有导航SR结果。

## 数据与对照

实际Revised TRAIN为21,878条，并非历史版本22,002条。全部进入清单，Cambridge14,259条用于新视觉模块拟合，Birmingham7,619条为城市/地图及物理地标隔离Dev。冻结Geometry/Relation此前训练过全TRAIN，因此这是新视觉模块留出确认，不是全模型未知地图泛化。

3,440组公开引用组合规划59,800张正射图像；每组包含完整地标上下文384px全景和同范围内128m/64m上限局部块，局部块数量中位15.5、最大54。156条无法获得有效Polygon仍计入分母、保留Geometry+Relation回退。566次非法Polygon片段被记录，不据GT补选地标。图像区域由全部合法公开引用及固定上下文决定，无目标坐标输入。

冻结对照为Geometry+Relation、起点真实RGB、完整地标全景、多尺度全景+局部。后三者采用相同公开引用接口；多引用平均文本与旧唯一Anchor门控运行时不同，不能混称原导航成绩。离线关闭追踪门控，目的是检验图像内容，正常导航不变。

完整地标和局部图像均为特权底图访问；局部图来自同一正射纹理，不是新视角或真实因果历史。按世界格平均重复证据。实际可见性及像素实例身份未知，世界R20、Top-K与投影覆盖均不代表实例识别准确率。

## 训练协议（Dev评分前固定）

冻结SigLIP2、BERT、Geometry、Relation、Trajectory/Stop及其他Head，仅训练PGVB和既有joint_logits。固定seed42、3轮、最终Checkpoint；不根据Dev挑轮数或阈值。PGVB学习率3e-5，joint1e-4，AdamW，batch4，梯度裁剪1。总视觉修正仍限制±0.5 logit，不扫描增益。

TRAIN目标15m软标签CE；仅在有效采样网格对GT20m范围有空间支持时启用定位损失，否则仅保真；保真KL权重0.2。该支持仅作训练损失权重，不参与裁剪或推理。无可靠目标框，不能声称像素实例监督。

联合Head能读取冻结几何Context，存在地图捷径风险。因此同时确认RGB置零、有效Patch内容错位、PGVB-only消融；只有真实RGB优于这些控制才作为视觉增益。特权Checkpoint采用独立格式，普通导航加载器拒绝加载。

## 已完成检查与恢复

预检371项CPU测试通过；新增多视图梯度、控件不原地修改、无观测回退、重复证据不累计、特权Checkpoint隔离等测试后，7项专项测试通过。真实GPU缓存Smoke16条完成：7组引用、89张特权图像、1条当前视野具有地标支持；这不是训练或性能结论。

服务器目录：`GraphNav/runs/complete_landmark_rgb_s42_20261105/`，tmux `complete_landmark_cache`、`complete_landmark_training`；逐组与逐指令原子保存，输入/代码SHA不符拒绝缓存续跑。训练保存模型、优化器、随机状态、epoch/cursor，支持恢复。冻结预测和训练后预测分别封存，然后统一读取Dev标签评分。

遵守PR18数据说明：许可未确认的原始底图裁剪不推GitHub，真实RGB图册留服务器；仓库保存采样几何图、Belief图、统计与HTML生成代码。

## 待完成

全量缓存、10步真实GPU梯度Smoke、冻结对照、3轮视觉训练、独立Dev一次确认、分层指标与可视化。没有完成的实验不列为成功。

## 第二阶段工程核验

- 修复局部视图仅与Polygon边缘接触导致零面积交集异常；6,924张已保存视图的形状Token逐项完全一致后，保留原缓存续跑。
- CPU Profiling14张图0.797s，原576格逐ROI构造占0.334s；改为只计算最终144格的等价向量化Mask。3种Yaw与随机有效像素条件下，world/pixels/masks/quality/valid逐元素完全相同，不改变正常预处理入口。
- 完整公开Polygon核查发现71组Cambridge引用的合法同名片段被旧实体容量截断。按同地图/同公开名称/同类型补齐，影响219条fit指令，已有2条相关缓存归档后重算。Dev输入逐项完全不变；从未按GT选择片段。
- 10步真实GPU训练Smoke通过，首步PGVB梯度范数和2.5495、融合Head0.2363；训练分支GPU峰值0.2023GiB（仅缓存特征训练，不含独立缓存进程）。早期缓存只有Dev时产生过零步空跑，已拒绝作为Smoke并增加空fit检查。
- 已尝试调用adversarial-review；辅助服务因workspace routing discovery timed out失败，不能列为独立审查通过。继续以明确的单元测试、真实GPU与输入一致性核验为证据。
