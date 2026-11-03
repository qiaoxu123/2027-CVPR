# 进度

2026-11-03：全量语言关联分析已推送2303ce1。创建独立Goal分支；实读Stage1、matcher、patch pooling与runtime。读取真实权重核对grid和gain，定位完整恢复缓存。尚未新训练、尚无新SR。

2026-11-03：新增read-only审计脚本及中文设计报告。使用恢复完整TRAIN缓存首个有支持帧，复制原matcher权重在RTX PRO6000上完成forward/backward（合成梯度检查目标，非训练），梯度有限。原始tests/test_stage1.py与tests/test_appearance_matching.py合计10 passed（1.29s）。原模型文件未写入，未启动导航。新分支目前仅审计/设计，没有声称新模块收益。

2026-11-03 第二阶段：独立FixedYawObservation实现且原入口未改。新增7项相关测试，连同原Stage1/appearance共17 passed。实际采集24物理实例/16地图/87帧，保存独立公开图像输入与private labels；原/小ROI两套冻结SigLIP2编码均完成。分离predict与score、先封存预测后读标签。原ROI22帧无候选，小ROI13帧；目标点深度支持但全帧无ROI从10降4。小ROI全文空间命中3→11/87，但shuffle也11，尚不能证明视觉有效；分尺度角色规则没有稳定优势，不据此扩大正式测试。生成24真实RGB案例图册；下一阶段需观测内细粒度位置头+独立TRAIN训练确认，不再调本24例固定权重。
