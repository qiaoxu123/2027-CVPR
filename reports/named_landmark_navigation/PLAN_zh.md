# 公开具名地标导航能力：冻结E1/E2预注册

2026-11-04。用户授权的独立Val-Seen地标任务诊断，不训练、不修改原模型或正常入口、不访问Test。资源预检使用get-available-resources技能；其NVIDIA扫描误报无GPU，因此以实际nvidia-smi及torch CUDA预检为准，不假称无GPU或停止其它任务。

## 固定任务

从前一报告1161条direct_geometry原指令中，仅提取hops=1路径引用的具名地标。根据公开CityRefer全名匹配，保留同名全部Polygon；名称对应多个碎片或实例时以其并集定义名称级目标，绝不利用原目标坐标消歧。

在起点距Polygon并集>20m的候选中，按名称组去重和seed0哈希选择原任务起点，再按地图轮转分层选64个不同map+name组：Building32、Road24、Other8，覆盖23地图，23任务有多个公开Polygon。Road包括TrafficRoad/Footpath/Bridge及其同名混合。完整候选351组；不是随机代表全部CityNav的新测试集，也不是全量性能结论。

统一文本`Navigate to <公开地标名称>.`。原指令只在标注侧用于来源追溯，不输入新导航任务；不提供原任务目标坐标、实例ID、Contour或专家轨迹。新任务仅按地标名称和原始合法起点定义。

## 两个主对照

- **E1**：每步原冻结模型完整执行，再将输出GoalField替换为公开Polygon给定Goal分布。Goal是起点到合法地图范围内Polygon并集最近点，不是道路质心；初始后固定，防止目标随移动追踪漂移。分布sigma20m，首候选坐标为精确连续Goal，其他候选及首候选上下文来自原模型相邻网格的dense_context；没有伪造视觉Query特征。此离散上下文/连续Goal桥接属于明确的Oracle接口干预，需记录误差。原学习式Trajectory/Stop、Guarded-Blend、BoundedWaypointController均保持原代码及权重。
- **E2**：完全不干预Goal，原冻结GraphNav-V1根据新地标名称指令与原视觉/历史分支进行真实闭环。重新计算BERT和SigLIP语言特征，不复用旧指令语言缓存。

两组起点、文本、相机机制、真实RGB渲染、权重和控制协议一致。沿用原生每步相机yaw，不施加最近实验的固定yaw封装，不改变预处理。Horizon20、水平50m、垂直10m、动作高度10–100m、Stop阈值0.3及原到达守卫不变。不是只走教师轨迹或离线回放。

## 评价

SR为结束位置到公开Polygon并集最短XY距离≤20m；OSR为轨迹任一位置≤20m；NE为最终Polygon距离。另记录精确进入Polygon、20m首次进入、停止类型、Stop概率/到达及高度守卫、逐步Goal到Polygon距离和Goal丢失。开始已≤20m不进入主64集，所有排除原因保存。

这种评价不同于原CityNav目标点20m，尤其长道路和大建筑并集更宽松，不能将SR与原目标任务直接比较。没有新任务专家路径，故不报告伪造的nDTW或官方SPL。

## 泄漏及接口审计

普通E2仅接收PublicEpisode中地图、名称指令、真实起点和具名参考名称；E1公开目标仅在专用Oracle hook使用，普通入口无新增flag/代码修改。原目标GT不参与任务选择、Goal生成或在线决策。原数据加载侧包含的旧SR/GT不用于新任务采样。检查重名碎片、16地标截断、模型训练目标与“自身地标名称”任务分布差异，以及边界/上下文映射。

先运行两对smoke核验保存、真实GPU推理、Oracle接入与恢复；通过后同一冻结配置补齐64×2。支持逐Episode JSONL恢复，tmux持续运行。不依成功率更换文本、模型、Stop或样本。
