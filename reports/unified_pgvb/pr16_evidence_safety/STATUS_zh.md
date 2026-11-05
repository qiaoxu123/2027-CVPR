# PR #16 验证阶段

基准PR Head：b4584aadcf8e67c34f87b118d149c3b4a28b6297。

已检查当前帧RGB确认、多Anchor/多ID视觉拒绝、未校准方向回退，以及Stable Goal重规划拒绝原因。补齐测试夹具predicate字段；当前地图公开Catalog成为Polygon/ID的唯一来源，修复真实Catalog整数ID接口；回退后的有效Belief与被拒绝的原始Head输出分开记录。

TRAIN审计：573查询、1717缓存状态；467单公开ID、106同名多ID；现有任务文件未发现与当前公开地图轮廓错配。5条目标ID与Anchor ID重叠，50个目标点落在Anchor并集内。这些是标注/命名歧义代理，不能据GT挑选或删除推理Anchor。

同步north/south原文、结构ID及BERT/Geometry特征的63条探针未验证方向Head可靠性；未给改写指令伪造GT。TIF缺少CRS，不依据GT翻轴。保留未校准方向回退，不把回退当作学会方向。

保留原aa359894…Checkpoint，不盲目重训：当前证据指向语义/实例监督不充分，而不是训练步数不足。仅在TRAIN完成代码修复后冻结，再运行原56条任务三组闭环，不用Val调参，不访问Test。

Stable Goal固定阈值0.35/连续2帧；200帧CPU等价回放（含10次重规划）与PR前候选、坐标、Context及状态完全相同。Stop/Trajectory/Horizon不变。CPU与2条TRAIN真实闭环Smoke已执行；最终结果见REPORT_zh.md。

实际调用本地codex adversarial-review命令，但workspace routing discovery超时，没有生成有效独立审查结论。保存原始工具输出，不冒充审查通过。
