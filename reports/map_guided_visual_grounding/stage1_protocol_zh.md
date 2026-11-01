# 阶段一：协议审查

- GitHub 基点已核实为 f2faec2ef0089dd98049ff11ece51df0cc905143，新建独立实验分支。
- 现有1275 TRAIN/635内部Dev状态均能追溯到原始缓存manifest、当前384×384 PNG及900个AerialVG查询。正式数据不读取。
- renderer.current_rgbd使用地面平面四角单应裁剪正射RGB；x向东/y向北，footprint按像素TL/TR/BR/BL排列，yaw0图像上方为世界east。缓存包含pose、footprint_world、depth_valid，但不含depth_m、真实相机K/R/t或遮挡真值。
- 地图contour是世界米制；Stage1中的landmarks.contour为归一化坐标，必须先按world_bounds转换，不能与原CityRefer米制轮廓混用。完整公共具名地标可从CityRefer对象目录提取，仅传name/type/contour/position，不传目标ID或描述标注。
- 图像投影只能称几何覆盖（完整/部分/不覆盖），不能等同真实可见或实例分割。depth_valid只能过滤无效/相机以下表面支持，不能识别遮挡。
- 官方CityRefer对象目录有每个实例的object_id、contour、position、dimension，可与仅TRAIN源episode的object_ids关联，供独立评价端建立注册实例框IoU评价。GT目标坐标/对象ID不进入推理。
- 人工投影误差目前未知；后续提供核验图册和表单，不以单应回投影误差替代人工误差。物理建筑高度/视角遮挡不可从当前缓存恢复。

下一阶段优先实现投影、实际参考ROI冻结AerialVG特征提取及TRAIN可视化，再冻结设计开展635状态内部Dev诊断。不会训练网络。
